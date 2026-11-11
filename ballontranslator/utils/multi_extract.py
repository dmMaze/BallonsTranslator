"""Reconcile several detection+OCR passes of the same page.

A multi-pass extraction runs the same page through up to three detector/OCR
pairs, so it ends up with one block list per pass instead of a single one. This
module decides what the page looks like afterwards, following two rules chosen
for the workflow:

* Regions are the union of every pass. A block only one pass found is kept, so
  a detector miss never loses text, and it is reported as weakly supported.
* Text is never decided silently. The reference pass keeps the page as it is,
  and every block whose passes disagree is reported with all candidates so the
  user can pick. Merged output is therefore always reviewable.

Blocks are matched by bounding-box IoU, greedily, at most one match per pair of
passes, which keeps neighbouring bubbles from chaining into one group.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .textblock import TextBlock, sort_regions


@dataclass
class MergeConflict:
    """A merged region the user may want to look at."""

    block: TextBlock
    # Pass that supplied the geometry and the text currently on the block.
    chosen_pass: int
    # (pass_index, candidate block) for every other pass that read this region.
    candidates: List[Tuple[int, TextBlock]] = field(default_factory=list)
    # Passes that detected this region at all, in ascending order.
    support: Tuple[int, ...] = ()

    @property
    def single_pass(self) -> bool:
        """True when only one pass detected the region."""
        return len(self.support) == 1

    @property
    def texts(self) -> List[str]:
        """Every distinct non-empty reading, reference pass first."""
        readings = [self.block] + [candidate for _, candidate in self.candidates]
        texts = []
        for block in readings:
            text = block.get_text().strip()
            if text and text not in texts:
                texts.append(text)
        return texts

    @property
    def text_differs(self) -> bool:
        """True when the passes read different text for the same region."""
        return len(self.texts) > 1


@dataclass
class MergeResult:
    blocks: List[TextBlock]
    conflicts: List[MergeConflict]

    @property
    def weak_regions(self) -> List[MergeConflict]:
        """Regions only one pass detected."""
        return [conflict for conflict in self.conflicts if conflict.single_pass]

    @property
    def text_conflicts(self) -> List[MergeConflict]:
        """Regions the passes detected together but read differently."""
        return [
            conflict
            for conflict in self.conflicts
            if not conflict.single_pass and conflict.text_differs
        ]


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``[x, y, w, h]`` boxes.

    >>> box_iou([0, 0, 10, 10], [0, 0, 10, 10])
    1.0
    >>> box_iou([0, 0, 10, 10], [10, 0, 10, 10])
    0.0
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection_x = min(ax + aw, bx + bw) - max(ax, bx)
    intersection_y = min(ay + ah, by + bh) - max(ay, by)
    if intersection_x <= 0 or intersection_y <= 0:
        return 0.0
    intersection = intersection_x * intersection_y
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def group_regions(
    passes: Sequence[Sequence[TextBlock]],
    iou_threshold: float = 0.5,
) -> List[List[Tuple[int, int]]]:
    """Group blocks across passes that cover the same physical text region.

    Each group holds ``(pass_index, block_index)`` sorted by pass, and groups
    keep the order the first pass saw them in.

    >>> same = TextBlock(_bounding_rect=[0, 0, 10, 10])
    >>> [len(group) for group in group_regions([[same], [same]])]
    [2]
    """
    parent = []
    owner = []
    local_index = []
    boxes = []
    for pass_index, blocks in enumerate(passes):
        for block_index, block in enumerate(blocks):
            parent.append(len(parent))
            owner.append(pass_index)
            local_index.append(block_index)
            boxes.append(block.bounding_rect())

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pass_indexes = sorted(set(owner))
    for position, first in enumerate(pass_indexes):
        for second in pass_indexes[position + 1:]:
            edges = []
            for left in range(len(boxes)):
                if owner[left] != first:
                    continue
                for right in range(len(boxes)):
                    if owner[right] != second:
                        continue
                    score = box_iou(boxes[left], boxes[right])
                    if score >= iou_threshold:
                        edges.append((score, left, right))
            edges.sort(reverse=True)
            taken_left, taken_right = set(), set()
            for _, left, right in edges:
                if left in taken_left or right in taken_right:
                    continue
                taken_left.add(left)
                taken_right.add(right)
                parent[find(right)] = find(left)

    grouped = {}
    for index in range(len(boxes)):
        grouped.setdefault(find(index), []).append(index)

    regions = [
        sorted(
            ((owner[i], local_index[i]) for i in indexes),
            key=lambda member: member[0],
        )
        for indexes in grouped.values()
    ]
    # Deterministic order; the merged page is sorted again in merge_passes().
    regions.sort(key=lambda members: (members[0][0], members[0][1]))
    return regions


def merge_passes(
    passes: Sequence[Sequence[TextBlock]],
    iou_threshold: float = 0.5,
    reference_pass: int = 0,
) -> MergeResult:
    """Fold per-pass block lists into one page, keeping disagreements visible.

    >>> left = TextBlock(_bounding_rect=[0, 0, 10, 10])
    >>> left.text = ['あ']
    >>> same_spot = TextBlock(_bounding_rect=[0, 0, 11, 11])
    >>> same_spot.text = ['い']
    >>> extra = TextBlock(_bounding_rect=[80, 0, 10, 10])
    >>> extra.text = ['う']
    >>> result = merge_passes([[left], [same_spot, extra]])
    >>> sorted(block.get_text() for block in result.blocks)
    ['あ', 'う']
    >>> len(result.text_conflicts), len(result.weak_regions)
    (1, 1)
    """
    merged: List[TextBlock] = []
    conflicts: List[MergeConflict] = []

    for members in group_regions(passes, iou_threshold):
        reference = next(
            (member for member in members if member[0] == reference_pass),
            members[0],
        )
        chosen = deepcopy(passes[reference[0]][reference[1]])
        candidates = [
            (pass_index, deepcopy(passes[pass_index][block_index]))
            for pass_index, block_index in members
            if (pass_index, block_index) != reference
        ]

        if not chosen.get_text().strip():
            # An empty read is not a disagreement: take what another pass saw
            # rather than leaving a detected region blank.
            for _, candidate in candidates:
                if candidate.get_text().strip():
                    chosen.text = deepcopy(candidate.text)
                    break

        merged.append(chosen)
        conflicts.append(
            MergeConflict(
                block=chosen,
                chosen_pass=reference[0],
                candidates=candidates,
                support=tuple(pass_index for pass_index, _ in members),
            )
        )

    return MergeResult(blocks=sort_regions(merged), conflicts=conflicts)


def conflict_payload(conflict: MergeConflict) -> dict:
    """Attachable form of a conflict, stored on the block for the reviewer.

    Only the readings are kept: geometry always comes from the chosen pass, so
    a candidate never needs to bring its own box.

    >>> a, b = TextBlock(_bounding_rect=[0, 0, 10, 10]), TextBlock(_bounding_rect=[0, 0, 10, 10])
    >>> a.text, b.text = ['あ'], ['い']
    >>> payload = conflict_payload(merge_passes([[a], [b]]).conflicts[0])
    >>> payload['chosen'], payload['support'], payload['candidates']
    (0, [0, 1], [{'pass': 1, 'text': ['い']}])
    """
    return {
        'chosen': conflict.chosen_pass,
        'support': list(conflict.support),
        'candidates': [
            {'pass': pass_index, 'text': _as_lines(block.text)}
            for pass_index, block in conflict.candidates
        ],
    }


def _as_lines(text) -> List[str]:
    if isinstance(text, str):
        return [text]
    return [str(line) for line in text]


def apply_conflict_payloads(result: MergeResult) -> int:
    """Attach review payloads to the merged blocks; returns how many were marked.

    Regions every pass agreed on stay clean, so the reviewer only walks through
    the places a decision is actually needed. A region only one pass detected is
    marked as well: the union kept it, and the reviewer should know that.

    >>> a = TextBlock(_bounding_rect=[0, 0, 10, 10])
    >>> a.text = ['あ']
    >>> b = TextBlock(_bounding_rect=[0, 0, 10, 10])
    >>> b.text = ['い']
    >>> merged = merge_passes([[a], [b]])
    >>> apply_conflict_payloads(merged)
    1
    >>> has_conflict(merged.blocks[0])
    True
    """
    marked = 0
    for conflict in result.conflicts:
        if conflict.text_differs or conflict.single_pass:
            conflict.block.multi_extract = conflict_payload(conflict)
            marked += 1
        else:
            conflict.block.multi_extract = {}
    return marked


def has_conflict(block: TextBlock) -> bool:
    """True while a block still carries an unreviewed multi-pass payload.

    >>> has_conflict(TextBlock())
    False
    """
    return bool(getattr(block, 'multi_extract', None))


def iter_conflicts(pages) -> List[Tuple[str, TextBlock]]:
    """Every unreviewed block of a project, in page order.

    >>> pages = {'a.jpg': [TextBlock()]}
    >>> iter_conflicts(pages)
    []
    """
    found = []
    for page_name, blocks in pages.items():
        for block in blocks:
            if has_conflict(block):
                found.append((page_name, block))
    return found


def conflict_count(pages) -> int:
    """Number of unreviewed blocks in a project.

    >>> conflict_count({'a.jpg': [TextBlock(), TextBlock()]})
    0
    """
    return sum(1 for blocks in pages.values() for block in blocks if has_conflict(block))


def readings(block: TextBlock) -> List[Tuple[int, List[str]]]:
    """Readings a reviewer can choose from: the chosen pass first.

    >>> block = TextBlock()
    >>> block.text = ['あ']
    >>> block.multi_extract = {'chosen': 0, 'candidates': [{'pass': 1, 'text': ['い']}]}
    >>> readings(block)
    [(0, ['あ']), (1, ['い'])]
    """
    payload = getattr(block, 'multi_extract', None) or {}
    options = [(payload.get('chosen', 0), _as_lines(block.text))]
    for candidate in payload.get('candidates', []):
        options.append((candidate['pass'], _as_lines(candidate['text'])))
    return options


def resolve(block: TextBlock, pass_index: int) -> Tuple[List[str], dict]:
    """Apply one reading to the block and clear its conflict.

    Returns the previous text and payload so a caller can undo the change.
    Choosing the pass that already supplied the text only marks the block as
    reviewed.

    >>> block = TextBlock()
    >>> block.text = ['あ']
    >>> block.multi_extract = {'chosen': 0, 'candidates': [{'pass': 1, 'text': ['い']}]}
    >>> resolve(block, 1)[0]
    ['あ']
    >>> block.get_text(), has_conflict(block)
    ('い', False)
    """
    payload = deepcopy(getattr(block, 'multi_extract', None) or {})
    old_text = _as_lines(block.text)
    for candidate_pass, candidate_text in readings(block):
        if candidate_pass == pass_index:
            block.text = list(candidate_text)
            break
    block.multi_extract = {}
    return old_text, payload

