# coding: utf-8
from enum import Enum

import os
import xcffib as xcb
from PyQt6.QtCore import QPointF, Qt
from xcffib.xproto import (ButtonIndex, ButtonMask, ButtonReleaseEvent,
                           ClientMessageData, ClientMessageEvent, EventMask,
                           xprotoExtension)


class WindowMessage(Enum):
    """ Window message enum class """
    # refer to: https://specifications.freedesktop.org/wm-spec/1.1/x170.html
    _NET_WM_MOVERESIZE_SIZE_TOPLEFT = 0
    _NET_WM_MOVERESIZE_SIZE_TOP = 1
    _NET_WM_MOVERESIZE_SIZE_TOPRIGHT = 2
    _NET_WM_MOVERESIZE_SIZE_RIGHT = 3
    _NET_WM_MOVERESIZE_SIZE_BOTTOMRIGHT = 4
    _NET_WM_MOVERESIZE_SIZE_BOTTOM = 5
    _NET_WM_MOVERESIZE_SIZE_BOTTOMLEFT = 6
    _NET_WM_MOVERESIZE_SIZE_LEFT = 7
    _NET_WM_MOVERESIZE_MOVE = 8
    _NET_WM_MOVERESIZE_SIZE_KEYBOARD = 9
    _NET_WM_MOVERESIZE_MOVE_KEYBOARD = 10
    _NET_WM_MOVERESIZE_CANCEL = 11


class LinuxMoveResize:
    """ Tool class for moving and resizing window """

    moveResizeAtom = None

    @classmethod
    def sendButtonReleaseEvent(cls, window, globalPos):
        """ send button release event

        Parameters
        ----------
        window: QWidget
            window to be moved or resized

        globalPos: QPoint
            the global point of mouse release event
        """
        globalPos = QPointF(QPointF(globalPos) *
                            window.devicePixelRatio()).toPoint()
        pos = window.mapFromGlobal(globalPos)

        # open the connection to X server
        conn = xcb.connect(os.environ.get('DISPLAY'))
        windowId = int(window.winId())
        xproto = xprotoExtension(conn)

        # refer to: https://www.x.org/releases/X11R7.5/doc/libxcb/tutorial/
        event = ButtonReleaseEvent.synthetic(
            detail=ButtonIndex._1,
            time=xcb.CurrentTime,
            root=conn.get_setup().roots[0].root,
            event=windowId,
            child=xcb.NONE,
            root_x=globalPos.x(),
            root_y=globalPos.y(),
            event_x=pos.x(),
            event_y=pos.y(),
            state=ButtonMask._1,
            same_screen=True,
        )
        xproto.SendEvent(True, windowId, EventMask.ButtonRelease, event.pack())
        conn.flush()

    @classmethod
    def startSystemMoveResize(cls, window, globalPos, message):
        """ resize window

        Parameters
        ----------
        window: QWidget
            window to be moved or resized

        globalPos: QPoint
            the global point of mouse release event

        message: int
            window message
        """
        cls.sendButtonReleaseEvent(window, globalPos)

        globalPos = QPointF(QPointF(globalPos) *
                            window.devicePixelRatio()).toPoint()

        # open the connection to X server
        conn = xcb.connect(os.environ.get('DISPLAY'))
        xproto = xprotoExtension(conn)

        if not cls.moveResizeAtom:
            cls.moveResizeAtom = xproto.InternAtom(
                False, len("_NET_WM_MOVERESIZE"), "_NET_WM_MOVERESIZE").reply().atom

        union = ClientMessageData.synthetic([
            globalPos.x(),
            globalPos.y(),
            message,
            ButtonIndex._1,
            0
        ], "I"*5)
        event = ClientMessageEvent.synthetic(
            format=32,
            window=int(window.winId()),
            type=cls.moveResizeAtom,
            data=union
        )
        xproto.UngrabPointer(xcb.CurrentTime)
        xproto.SendEvent(
            False,
            conn.get_setup().roots[0].root,
            EventMask.SubstructureRedirect | EventMask.SubstructureNotify,
            event.pack()
        )
        conn.flush()

    @classmethod
    def startSystemMove(cls, window, globalPos):
        """ move window """
        cls.startSystemMoveResize(
            window, globalPos, WindowMessage._NET_WM_MOVERESIZE_MOVE.value)

    @classmethod
    def starSystemResize(cls, window, globalPos, edges):
        """ resize window

        Parameters
        ----------
        window: QWidget
            window

        globalPos: QPoint
            the global point of mouse release event

        edges: `Qt.Edges`
            window edges
        """
        if not edges:
            return

        messageMap = {
            Qt.Edge.TopEdge: WindowMessage._NET_WM_MOVERESIZE_SIZE_TOP,
            Qt.Edge.TopEdge | Qt.Edge.LeftEdge: WindowMessage._NET_WM_MOVERESIZE_SIZE_TOPLEFT,
            Qt.Edge.TopEdge | Qt.Edge.RightEdge: WindowMessage._NET_WM_MOVERESIZE_SIZE_TOPRIGHT,
            Qt.Edge.BottomEdge: WindowMessage._NET_WM_MOVERESIZE_SIZE_BOTTOM,
            Qt.Edge.BottomEdge | Qt.Edge.LeftEdge: WindowMessage._NET_WM_MOVERESIZE_SIZE_BOTTOMLEFT,
            Qt.Edge.BottomEdge | Qt.Edge.RightEdge: WindowMessage._NET_WM_MOVERESIZE_SIZE_BOTTOMRIGHT,
            Qt.Edge.LeftEdge: WindowMessage._NET_WM_MOVERESIZE_SIZE_LEFT,
            Qt.Edge.RightEdge: WindowMessage._NET_WM_MOVERESIZE_SIZE_RIGHT,
        }
        cls.startSystemMoveResize(window, globalPos, messageMap[edges].value)
