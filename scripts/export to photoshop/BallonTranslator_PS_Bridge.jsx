// BallonTranslator_PS_Bridge.jsx
// Version: 2.5.1
// Multilingual Bidirectional Bridge between BallonsTranslator and Adobe Photoshop (CC 2019 - CC 2026+)

#target photoshop

var BT_BRIDGE_VERSION = "2.5.1";






// ==========================================
// Embedded JSON2 Parser / Serializer
// ==========================================
if (typeof JSON !== "object") {
    JSON = {};
}
(function () {
    "use strict";
    var rx_one = /^[\],:{}\s]*$/,
        rx_two = /\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g,
        rx_three = /"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g,
        rx_four = /(?:^|:|,)(?:\s*\[)+/g,
        rx_escapable = /[\\"\u0000-\u001f\u007f-\u009f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g;

    function quote(string) {
        rx_escapable.lastIndex = 0;
        return rx_escapable.test(string) ?
            "\"" + string.replace(rx_escapable, function (a) {
                var c = {
                    "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r",
                    "\"": "\\\"", "\\": "\\\\"
                }[a];
                return typeof c === "string" ? c : "\\u" + ("0000" + a.charCodeAt(0).toString(16)).slice(-4);
            }) + "\"" :
            "\"" + string + "\"";
    }

    function str(key, holder) {
        var i, k, v, length, mind = "", partial, value = holder[key];
        if (value && typeof value === "object" && typeof value.toJSON === "function") {
            value = value.toJSON(key);
        }
        switch (typeof value) {
            case "string": return quote(value);
            case "number": return isFinite(value) ? String(value) : "null";
            case "boolean":
            case "null": return String(value);
            case "object":
                if (!value) return "null";
                partial = [];
                if (Object.prototype.toString.apply(value) === "[object Array]") {
                    length = value.length;
                    for (i = 0; i < length; i += 1) {
                        partial[i] = str(i, value) || "null";
                    }
                    return "[" + partial.join(",") + "]";
                }
                for (k in value) {
                    if (Object.prototype.hasOwnProperty.call(value, k)) {
                        v = str(k, value);
                        if (v) {
                            partial.push(quote(k) + ":" + v);
                        }
                    }
                }
                return "{" + partial.join(",") + "}";
        }
    }

    if (typeof JSON.stringify !== "function") {
        JSON.stringify = function (value) {
            return str("", { "": value });
        };
    }

    if (typeof JSON.parse !== "function") {
        JSON.parse = function (text) {
            text = String(text);
            if (rx_one.test(text.replace(rx_two, "@").replace(rx_three, "]").replace(rx_four, ""))) {
                return eval("(" + text + ")");
            }
            throw new SyntaxError("JSON.parse error");
        };
    }
}());

// ==========================================
// Multilingual Localization Table (Unicode Escapes for 100% Reliability)
// ==========================================
var BT_I18N = {
    currentLang: "en",
    
    // Detect system or Photoshop language
    detectLanguage: function() {
        var loc = "";
        try {
            if (typeof app !== "undefined" && app.locale) {
                loc = String(app.locale).toLowerCase();
            } else if (typeof $ !== "undefined" && $.locale) {
                loc = String($.locale).toLowerCase();
            }
        } catch (e) {}
        if (loc.indexOf("ru") !== -1) return "ru";
        if (loc.indexOf("zh") !== -1 || loc.indexOf("cn") !== -1) return "zh";
        return "en";
    },

    strings: {
        dlgTitle: {
            en: "BallonsTranslator - Photoshop Bridge",
            ru: "\u041c\u043e\u0441\u0442 BallonsTranslator - Photoshop",
            zh: "BallonsTranslator - Photoshop \u6865\u63a5"
        },
        projInfo: {
            en: "Project Information",
            ru: "\u0418\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u044f \u043e \u043f\u0440\u043e\u0435\u043a\u0442\u0435",
            zh: "\u9879\u76ee\u4fe1\u606f"
        },
        projFile: {
            en: "Project File: ",
            ru: "\u0424\u0430\u0439\u043b \u043f\u0440\u043e\u0435\u043a\u0442\u0430: ",
            zh: "\u9879\u76ee\u6587\u4ef6: "
        },
        projDir: {
            en: "Directory: ",
            ru: "\u041f\u0430\u043f\u043a\u0430: ",
            zh: "\u76ee\u5f55: "
        },
        totalPages: {
            en: "Total Pages: ",
            ru: "\u0412\u0441\u0435\u0433\u043e \u0441\u0442\u0440\u0430\u043d\u0438\u0446: ",
            zh: "\u603b\u9875\u6570: "
        },
        selectPages: {
            en: "Select Pages to Import",
            ru: "\u0412\u044b\u0431\u043e\u0440 \u0441\u0442\u0440\u0430\u043d\u0438\u0446 \u0434\u043b\u044f \u0438\u043c\u043f\u043e\u0440\u0442\u0430",
            zh: "\u9009\u62e9\u8981\u5bfc\u5165\u7684\u9875\u9762"
        },
        selectAll: {
            en: "Select All",
            ru: "\u0412\u044b\u0431\u0440\u0430\u0442\u044c \u0432\u0441\u0435",
            zh: "\u5168\u9009"
        },
        deselectAll: {
            en: "Deselect All",
            ru: "\u0421\u043d\u044f\u0442\u044c \u0432\u0441\u0435",
            zh: "\u53d6\u6d88\u5168\u9009"
        },
        importOpts: {
            en: "Import & Styling Options",
            ru: "\u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438 \u0438\u043c\u043f\u043e\u0440\u0442\u0430 \u0438 \u0441\u0442\u0438\u043b\u0435\u0439",
            zh: "\u5bfc\u5165\u4e0e\u6837\u5f0f\u9009\u9879"
        },
        chkInpaint: {
            en: "Include Inpainted clean plate (/inpainted/)",
            ru: "\u0414\u043e\u0431\u0430\u0432\u0438\u0442\u044c \u0441\u043b\u043e\u0439 \u043a\u043b\u0438\u043d\u0430 (/inpainted/)",
            zh: "\u5305\u542b\u4fee\u56fe\u6e05\u6d01\u56fe\u5c42 (/inpainted/)"
        },
        chkMask: {
            en: "Include Text Mask layer (/mask/) [Hidden]",
            ru: "\u0414\u043e\u0431\u0430\u0432\u0438\u0442\u044c \u0441\u043b\u043e\u0439 \u043c\u0430\u0441\u043a\u0438 (/mask/) [\u0441\u043a\u0440\u044b\u0442]",
            zh: "\u5305\u542b\u6587\u672c\u8499\u7248\u56fe\u5c42 (/mask/) [\u9690\u85cf]"
        },
        chkTrans: {
            en: "Create Translation Text Layers",
            ru: "\u0421\u043e\u0437\u0434\u0430\u0442\u044c \u0442\u0435\u043a\u0441\u0442\u043e\u0432\u044b\u0435 \u0441\u043b\u043e\u0438 \u043f\u0435\u0440\u0435\u0432\u043e\u0434\u0430",
            zh: "\u521b\u5efa\u7ffb\u8bd1\u6587\u672c\u56fe\u5c42"
        },
        chkOrig: {
            en: "Create Original OCR Reference Layers [Hidden]",
            ru: "\u0421\u043e\u0437\u0434\u0430\u0442\u044c \u0441\u043b\u043e\u0438 \u043e\u0440\u0438\u0433\u0438\u043d\u0430\u043b\u044c\u043d\u043e\u0433\u043e OCR [\u0441\u043a\u0440\u044b\u0442]",
            zh: "\u521b\u5efa\u539f\u6587OCR\u53c2\u8003\u56fe\u5c42 [\u9690\u85cf]"
        },
        chkSmartBox: {
            en: "Smart Balloon Bounds (Adapt box for horizontal text)",
            ru: "\u0423\u043c\u043d\u044b\u0435 \u0433\u0440\u0430\u043d\u0438\u0446\u044b (\u0430\u0434\u0430\u043f\u0442\u0430\u0446\u0438\u044f \u043f\u043e\u0434 \u0433\u043e\u0440\u0438\u0437\u043e\u043d\u0442\u0430\u043b\u044c\u043d\u044b\u0439 \u0442\u0435\u043a\u0441\u0442)",
            zh: "\u667a\u80fd\u6c14\u6ce1\u8fb9\u754c (\u9002\u5e94\u6a2a\u6392\u6587\u672c)"
        },
        chkStroke: {
            en: "Smart Stroke (Auto-add 3px white outline on text)",
            ru: "\u0411\u0435\u043b\u0430\u044f \u043e\u0431\u0432\u043e\u0434\u043a\u0430 (\u0430\u0432\u0442\u043e-3px \u0434\u043b\u044f \u0447\u0438\u0442\u0430\u0435\u043c\u043e\u0441\u0442\u0438 \u043d\u0430 \u0441\u043a\u0440\u0438\u043d\u0442\u043e\u043d\u0430\u0445)",
            zh: "\u667a\u80fd\u767d\u8272\u63cf\u8fb9 (\u81ea\u52a83px\u767d\u8fb9\u63d0\u5347\u53ef\u8bfb\u6027)"
        },
        chkVCenter: {
            en: "Vertical Centering (Auto-center text vertically & fit frame)",
            ru: "\u0412\u0435\u0440\u0442\u0438\u043a\u0430\u043b\u044c\u043d\u043e\u0435 \u0446\u0435\u043d\u0442\u0440\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435 (\u0430\u0432\u0442\u043e-\u0446\u0435\u043d\u0442\u0440 \u0438 \u043f\u043e\u0434\u0433\u043e\u043d\u043a\u0430 \u0440\u0430\u043c\u043a\u0438)",
            zh: "\u5782\u76f4\u5c45\u4e2d (\u81ea\u52a8\u5782\u76f4\u5c45\u4e2d\u5e76\u88c1\u526a\u6587\u672c\u6846)"
        },
        btnSavePSD: {
            en: "Save PSD to JSON",
            ru: "\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c PSD \u0432 JSON",
            zh: "\u4fdd\u5b58PSD\u5230JSON"
        },
        btnImport: {
            en: "Import Selected Pages",
            ru: "\u0418\u043c\u043f\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u044b",
            zh: "\u5bfc\u5165\u6240\u9009\u9875\u9762"
        },
        btnCancel: {
            en: "Cancel",
            ru: "\u041e\u0442\u043c\u0435\u043d\u0430",
            zh: "\u53d6\u6d88"
        },
        importSuccess: {
            en: "Successfully imported {count} page(s) into Photoshop!",
            ru: "\u0423\u0441\u043f\u0435\u0448\u043d\u043e \u0438\u043c\u043f\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u043e {count} \u0441\u0442\u0440\u0430\u043d\u0438\u0446 \u0432 Photoshop!",
            zh: "\u6210\u529f\u5bfc\u5165 {count} \u9875\u5230 Photoshop\uff01"
        },
        saveSuccess: {
            en: "Successfully saved {count} text blocks back to page '{page}' in project JSON!",
            ru: "\u0423\u0441\u043f\u0435\u0448\u043d\u043e \u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u043e {count} \u0431\u043b\u043e\u043a\u043e\u0432 \u0434\u043b\u044f \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u044b '{page}' \u0432 JSON!",
            zh: "\u6210\u529f\u4fdd\u5b58 {count} \u4e2a\u6587\u672c\u5757\u5230 '{page}' \u9875\u7684\u9879\u76eeJSON\uff01"
        }
    },

    t: function(key, params) {
        var lang = this.currentLang;
        var entry = this.strings[key];
        var text = (entry && entry[lang]) ? entry[lang] : (entry && entry["en"] ? entry["en"] : key);
        if (params) {
            for (var p in params) {
                if (params.hasOwnProperty(p)) {
                    text = text.replace(new RegExp("\\{" + p + "\\}", "g"), params[p]);
                }
            }
        }
        return text;
    }
};

// Initialize locale
BT_I18N.currentLang = BT_I18N.detectLanguage();

// ==========================================
// Photoshop ActionManager Helpers (Stroke FX, etc.)
// ==========================================
var BT_PS = (function () {
    var s2t = stringIDToTypeID;

    function applyStrokeFX(layer, strokeWidthPx, r, g, b, opacity, position) {
        if (!strokeWidthPx || strokeWidthPx <= 0) return;
        try {
            app.activeDocument.activeLayer = layer;
            var posKey = "OutF"; // Outside
            if (position === "inside") posKey = "InsF";
            else if (position === "center") posKey = "CtrF";

            var desc = new ActionDescriptor();
            var ref = new ActionReference();
            ref.putProperty(s2t("property"), s2t("layerEffects"));
            ref.putEnumerated(s2t("layer"), s2t("ordinal"), s2t("targetEnum"));
            desc.putReference(s2t("null"), ref);

            var strokeDesc = new ActionDescriptor();
            strokeDesc.putBoolean(s2t("enabled"), true);
            strokeDesc.putBoolean(s2t("present"), true);
            strokeDesc.putBoolean(s2t("showInDialog"), true);
            strokeDesc.putEnumerated(s2t("style"), s2t("frameStyle"), s2t(posKey));
            strokeDesc.putEnumerated(s2t("paintType"), s2t("fillFrameStyle"), s2t("solidColorLayer"));

            var colorDesc = new ActionDescriptor();
            colorDesc.putDouble(s2t("red"), (r !== undefined) ? r : 255);
            colorDesc.putDouble(s2t("grain"), (g !== undefined) ? g : 255);
            colorDesc.putDouble(s2t("blue"), (b !== undefined) ? b : 255);
            strokeDesc.putObject(s2t("color"), s2t("RGBColor"), colorDesc);

            strokeDesc.putUnitDouble(s2t("size"), s2t("pixelsUnit"), strokeWidthPx);
            strokeDesc.putUnitDouble(s2t("opacity"), s2t("percentUnit"), (opacity !== undefined) ? opacity : 100);

            var lefxDesc = new ActionDescriptor();
            lefxDesc.putObject(s2t("frameFX"), s2t("frameFX"), strokeDesc);
            desc.putObject(s2t("to"), s2t("layerEffects"), lefxDesc);

            executeAction(s2t("set"), desc, DialogModes.NO);
        } catch (e) {}
    }

    var fontCache = null;
    function initFontCache() {
        if (fontCache !== null) return fontCache;
        fontCache = {
            byPostScript: {},
            byFamily: {},
            all: []
        };
        try {
            var total = app.fonts.length;
            for (var i = 0; i < total; i++) {
                var f = app.fonts[i];
                var psName = f.postScriptName;
                var family = f.family.toLowerCase();
                fontCache.byPostScript[psName.toLowerCase()] = psName;
                if (!fontCache.byFamily[family]) {
                    fontCache.byFamily[family] = [];
                }
                fontCache.byFamily[family].push(psName);
                fontCache.all.push(psName);
            }
        } catch (e) {}
        return fontCache;
    }

    function resolveFontPostScript(requestedName, isBold, isItalic) {
        if (!requestedName) return null;
        var cache = initFontCache();
        var req = requestedName.toLowerCase().replace(/[\s\-_]+/g, "");

        for (var ps in cache.byPostScript) {
            var cleanPS = ps.replace(/[\s\-_]+/g, "");
            if (cleanPS === req) return cache.byPostScript[ps];
        }

        for (var fam in cache.byFamily) {
            var cleanFam = fam.replace(/[\s\-_]+/g, "");
            if (cleanFam === req || cleanFam.indexOf(req) !== -1 || req.indexOf(cleanFam) !== -1) {
                var variants = cache.byFamily[fam];
                if (isBold && isItalic) {
                    for (var b = 0; b < variants.length; b++) {
                        var v = variants[b].toLowerCase();
                        if (v.indexOf("bold") !== -1 && (v.indexOf("italic") !== -1 || v.indexOf("oblique") !== -1)) return variants[b];
                    }
                }
                if (isBold) {
                    for (var b2 = 0; b2 < variants.length; b2++) {
                        if (variants[b2].toLowerCase().indexOf("bold") !== -1) return variants[b2];
                    }
                }
                if (isItalic) {
                    for (var it = 0; it < variants.length; it++) {
                        var vit = variants[it].toLowerCase();
                        if (vit.indexOf("italic") !== -1 || vit.indexOf("oblique") !== -1) return variants[it];
                    }
                }
                return variants[0];
            }
        }
        return null;
    }

    function isUserCancelled(e) {
        if (!e) return false;
        if (e.number === 8007 || e.number === -128) return true;
        var msg = (e.message || String(e)).toLowerCase();
        return msg.indexOf("cancel") !== -1 || msg.indexOf("отмен") !== -1;
    }

    function openSilent(fileObj) {
        try {
            var desc = new ActionDescriptor();
            desc.putPath(s2t("null"), fileObj);
            executeAction(s2t("open"), desc, DialogModes.NO);
            return app.activeDocument;
        } catch (e) {
            if (isUserCancelled(e)) return null;
            try {
                return app.open(fileObj);
            } catch (e2) {
                if (isUserCancelled(e2)) return null;
                throw e2;
            }
        }
    }

    return {
        applyStrokeFX: applyStrokeFX,
        resolveFontPostScript: resolveFontPostScript,
        openSilent: openSilent,
        isUserCancelled: isUserCancelled
    };
})();

// Helper to sanitize path decoded string
function getCleanPath(f) {
    if (!f) return "";
    var p = f.fsName || f.fullName || f.name || String(f);
    try {
        p = decodeURI(p);
    } catch (e) {}
    return p;
}

// Find existing image file on disk safely
function findFile(baseDir, relativeName) {
    var cleanBase = getCleanPath(baseDir);
    var target = new File(cleanBase + "/" + relativeName);
    if (target.exists) return target;

    // Try without URL-encoding
    target = new File(cleanBase + "/" + decodeURI(relativeName));
    if (target.exists) return target;

    // Try alternate extensions
    var rawName = relativeName.replace(/\.[^\.]+$/, "");
    var exts = [".jpg", ".png", ".webp", ".jpeg", ".bmp", ".tif"];
    for (var i = 0; i < exts.length; i++) {
        var alt = new File(cleanBase + "/" + rawName + exts[i]);
        if (alt.exists) return alt;
    }
    return null;
}

// ==========================================
// Main Bridge Controller
// ==========================================
function runBallonTranslatorBridge() {
    // 1. Open project JSON (Check for Bridge Context from BallonsTranslator first)
    var jsonFile = null;
    var initialActivePage = null;
    try {
        var tempFolder = Folder.temp;
        var ctxFile = new File(tempFolder.fsName + "/bt_ps_bridge_context.json");
        if (ctxFile.exists) {
            ctxFile.encoding = "UTF-8";
            ctxFile.open("r");
            var ctxText = ctxFile.read();
            ctxFile.close();
            if (ctxText) {
                var ctx = JSON.parse(ctxText);
                if (ctx && ctx.project_path) {
                    var candFile = new File(ctx.project_path);
                    if (candFile.exists) {
                        var ageSec = (new Date().getTime() / 1000) - (ctx.timestamp || 0);
                        if (ageSec < 300) { // Valid within 5 minutes
                            jsonFile = candFile;
                            if (ctx.active_page) {
                                initialActivePage = ctx.active_page;
                            }
                        }
                    }
                }
            }
            try { ctxFile.remove(); } catch (rmErr) {}
        }
    } catch (ctxErr) {}

    if (!jsonFile || !jsonFile.exists) {
        jsonFile = File.openDialog(
            BT_I18N.t("selectJsonPrompt") || "Select BallonsTranslator Project JSON (*.json)",
            "JSON Files:*.json;All Files:*.*"
        );
    }
    if (!jsonFile || !jsonFile.exists) return;

    var projectDir = jsonFile.parent;
    var projectDirClean = getCleanPath(projectDir);

    jsonFile.encoding = "UTF-8";
    jsonFile.open("r");
    var rawText = jsonFile.read();
    jsonFile.close();

    var projectData;
    try {
        projectData = JSON.parse(rawText);
    } catch (err) {
        alert("Failed to parse project JSON:\n" + err.message, "Error", true);
        return;
    }

    if (!projectData.pages) {
        alert("Invalid project format: 'pages' key not found in JSON.", "Error", true);
        return;
    }

    var pageNames = [];
    for (var p in projectData.pages) {
        if (projectData.pages.hasOwnProperty(p)) {
            pageNames.push(p);
        }
    }
    if (pageNames.length === 0) {
        alert("No pages found in project.", "Warning");
        return;
    }

    // 2. Modern ScriptUI Dialog
    var dlg = new Window("dialog", BT_I18N.t("dlgTitle"));
    dlg.orientation = "column";
    dlg.alignChildren = ["fill", "top"];
    dlg.spacing = 8;
    dlg.margins = 14;

    // Top Language Switcher Bar
    var grpLang = dlg.add("group");
    grpLang.orientation = "row";
    grpLang.alignment = ["right", "top"];
    grpLang.add("statictext", undefined, "Language / \u042f\u0437\u044b\u043a:");
    var cmbLang = grpLang.add("dropdownlist", undefined, ["English", "\u0420\u0443\u0441\u0441\u043a\u0438\u0439", "\u4e2d\u6587"]);
    if (BT_I18N.currentLang === "ru") cmbLang.selection = 1;
    else if (BT_I18N.currentLang === "zh") cmbLang.selection = 2;
    else cmbLang.selection = 0;

    // Project Info Panel
    var pnlInfo = dlg.add("panel", undefined, BT_I18N.t("projInfo"));
    pnlInfo.orientation = "column";
    pnlInfo.alignChildren = ["left", "top"];
    pnlInfo.margins = 8;
    
    var decodedJsonName = decodeURI(jsonFile.name);
    var lblProj = pnlInfo.add("statictext", undefined, BT_I18N.t("projFile") + decodedJsonName);
    var lblDir = pnlInfo.add("statictext", undefined, BT_I18N.t("projDir") + projectDirClean);
    var lblTotal = pnlInfo.add("statictext", undefined, BT_I18N.t("totalPages") + pageNames.length);

    // Page Selection Panel
    var pnlPages = dlg.add("panel", undefined, BT_I18N.t("selectPages"));
    pnlPages.orientation = "column";
    pnlPages.alignChildren = ["fill", "top"];
    pnlPages.margins = 8;

    var listPages = pnlPages.add("listbox", undefined, [], { multiselect: true });
    listPages.preferredSize = [460, 160];
    var matchedInitial = false;
    for (var i = 0; i < pageNames.length; i++) {
        var pName = pageNames[i];
        var blkCount = (projectData.pages[pName] && projectData.pages[pName].length) ? projectData.pages[pName].length : 0;
        var item = listPages.add("item", pName + " (" + blkCount + " blocks)");
        item.pageKey = pName;
        if (initialActivePage && pName === initialActivePage) {
            item.selected = true;
            matchedInitial = true;
        }
    }
    if (!matchedInitial) {
        for (var sInit = 0; sInit < listPages.items.length; sInit++) {
            listPages.items[sInit].selected = true;
        }
    }

    var grpSelectBtns = pnlPages.add("group");
    grpSelectBtns.orientation = "row";
    var btnSelectAll = grpSelectBtns.add("button", undefined, BT_I18N.t("selectAll"));
    var btnSelectNone = grpSelectBtns.add("button", undefined, BT_I18N.t("deselectAll"));
    btnSelectAll.onClick = function () {
        for (var k = 0; k < listPages.items.length; k++) listPages.items[k].selected = true;
    };
    btnSelectNone.onClick = function () {
        for (var k = 0; k < listPages.items.length; k++) listPages.items[k].selected = false;
    };

    // Options Panel
    var pnlOpts = dlg.add("panel", undefined, BT_I18N.t("importOpts"));
    pnlOpts.orientation = "column";
    pnlOpts.alignChildren = ["left", "top"];
    pnlOpts.margins = 8;

    var chkInpaint = pnlOpts.add("checkbox", undefined, BT_I18N.t("chkInpaint"));
    chkInpaint.value = true;

    var chkMask = pnlOpts.add("checkbox", undefined, BT_I18N.t("chkMask"));
    chkMask.value = true;

    var chkTrans = pnlOpts.add("checkbox", undefined, BT_I18N.t("chkTrans"));
    chkTrans.value = true;

    var chkOrig = pnlOpts.add("checkbox", undefined, BT_I18N.t("chkOrig"));
    chkOrig.value = false;

    var chkSmartBox = pnlOpts.add("checkbox", undefined, BT_I18N.t("chkSmartBox"));
    chkSmartBox.value = true;

    var chkStroke = pnlOpts.add("checkbox", undefined, BT_I18N.t("chkStroke"));
    chkStroke.value = true;

    var chkVCenter = pnlOpts.add("checkbox", undefined, BT_I18N.t("chkVCenter"));
    chkVCenter.value = true;

    // Action Buttons
    var grpActions = dlg.add("group");
    grpActions.orientation = "row";
    grpActions.alignment = ["fill", "center"];

    var btnExportBack = grpActions.add("button", undefined, BT_I18N.t("btnSavePSD"));
    var btnCancel = grpActions.add("button", undefined, BT_I18N.t("btnCancel"), { name: "cancel" });
    var btnImport = grpActions.add("button", undefined, BT_I18N.t("btnImport"), { name: "ok" });

    // Dynamic UI re-translation handler
    cmbLang.onChange = function() {
        if (cmbLang.selection.index === 1) BT_I18N.currentLang = "ru";
        else if (cmbLang.selection.index === 2) BT_I18N.currentLang = "zh";
        else BT_I18N.currentLang = "en";

        dlg.text = BT_I18N.t("dlgTitle");
        pnlInfo.text = BT_I18N.t("projInfo");
        lblProj.text = BT_I18N.t("projFile") + decodedJsonName;
        lblDir.text = BT_I18N.t("projDir") + projectDirClean;
        lblTotal.text = BT_I18N.t("totalPages") + pageNames.length;
        pnlPages.text = BT_I18N.t("selectPages");
        btnSelectAll.text = BT_I18N.t("selectAll");
        btnSelectNone.text = BT_I18N.t("deselectAll");
        pnlOpts.text = BT_I18N.t("importOpts");
        chkInpaint.text = BT_I18N.t("chkInpaint");
        chkMask.text = BT_I18N.t("chkMask");
        chkTrans.text = BT_I18N.t("chkTrans");
        chkOrig.text = BT_I18N.t("chkOrig");
        chkSmartBox.text = BT_I18N.t("chkSmartBox");
        chkStroke.text = BT_I18N.t("chkStroke");
        chkVCenter.text = BT_I18N.t("chkVCenter");
        btnExportBack.text = BT_I18N.t("btnSavePSD");
        btnCancel.text = BT_I18N.t("btnCancel");
        btnImport.text = BT_I18N.t("btnImport");
    };

    // Reverse Sync handler
    btnExportBack.onClick = function () {
        if (!app.documents.length) {
            alert("No active document open in Photoshop to export.", "Warning");
            return;
        }
        var activeDoc = app.activeDocument;
        var activeDocName = decodeURI(activeDoc.name).replace(/\.[^\.]+$/, "");
        
        var matchedPage = null;
        for (var p = 0; p < pageNames.length; p++) {
            var rawPageName = decodeURI(pageNames[p]).replace(/\.[^\.]+$/, "");
            if (activeDocName === rawPageName || activeDoc.name === pageNames[p]) {
                matchedPage = pageNames[p];
                break;
            }
        }
        if (!matchedPage) {
            matchedPage = prompt("Document name doesn't match any page in JSON.\nEnter target page filename (e.g. " + pageNames[0] + "):", pageNames[0]);
            if (!matchedPage || !projectData.pages[matchedPage]) {
                alert("Export cancelled: Page not found in project.", "Warning");
                return;
            }
        }

        var transGroup = null;
        for (var g = 0; g < activeDoc.layerSets.length; g++) {
            if (activeDoc.layerSets[g].name.indexOf("Translations") !== -1) {
                transGroup = activeDoc.layerSets[g];
                break;
            }
        }

        var textLayers = [];
        if (transGroup) {
            for (var l = 0; l < transGroup.artLayers.length; l++) {
                if (transGroup.artLayers[l].kind === LayerKind.TEXT) {
                    textLayers.push(transGroup.artLayers[l]);
                }
            }
        } else {
            for (var l2 = 0; l2 < activeDoc.artLayers.length; l2++) {
                if (activeDoc.artLayers[l2].kind === LayerKind.TEXT) {
                    textLayers.push(activeDoc.artLayers[l2]);
                }
            }
        }

        if (textLayers.length === 0) {
            alert("No text layers found to export back to project.", "Notice");
            return;
        }

        var pageBlocks = projectData.pages[matchedPage] || [];
        var updatedCount = 0;

        for (var t = 0; t < textLayers.length; t++) {
            var tLayer = textLayers[t];
            var contents = tLayer.textItem.contents;
            if (t < pageBlocks.length) {
                pageBlocks[t].translation = contents;
                if (pageBlocks[t].fontformat) {
                    try {
                        pageBlocks[t].fontformat.font_size = tLayer.textItem.size.as("px");
                    } catch (e) {}
                }
                updatedCount++;
            }
        }

        projectData.pages[matchedPage] = pageBlocks;

        try {
            jsonFile.encoding = "UTF-8";
            jsonFile.open("w");
            jsonFile.write(JSON.stringify(projectData, null, 2));
            jsonFile.close();
            alert(BT_I18N.t("saveSuccess", { count: updatedCount, page: matchedPage }), "Success");
        } catch (saveErr) {
            alert("Failed to write to JSON:\n" + saveErr.message, "Error", true);
        }
    };

    if (dlg.show() !== 1) return;

    // Collect all selected pages properly from multiselect listbox
    var selectedPages = [];
    if (listPages.selection !== null) {
        if (listPages.selection instanceof Array) {
            for (var selI = 0; selI < listPages.selection.length; selI++) {
                if (listPages.selection[selI] && listPages.selection[selI].pageKey) {
                    selectedPages.push(listPages.selection[selI].pageKey);
                }
            }
        } else if (listPages.selection.pageKey) {
            selectedPages.push(listPages.selection.pageKey);
        }
    }
    if (selectedPages.length === 0) {
        for (var s = 0; s < listPages.items.length; s++) {
            if (listPages.items[s].selected) {
                selectedPages.push(listPages.items[s].pageKey);
            }
        }
    }

    if (selectedPages.length === 0) {
        alert("No pages selected for import.", "Notice");
        return;
    }

    // 3. Process each selected page
    var origRuler = app.preferences.rulerUnits;
    var origType = app.preferences.typeUnits;
    var origDialogs = app.displayDialogs;

    app.preferences.rulerUnits = Units.PIXELS;
    app.preferences.typeUnits = TypeUnits.PIXELS;
    app.displayDialogs = DialogModes.NO;

    var processedCount = 0;

    try {
        for (var idx = 0; idx < selectedPages.length; idx++) {
            var pageName = selectedPages[idx];
            var pageBlocks = projectData.pages[pageName] || [];

            var rawImgPath = findFile(projectDirClean, pageName);
            var inpaintImgPath = findFile(projectDirClean + "/inpainted", pageName);
            var maskImgPath = findFile(projectDirClean + "/mask", pageName);

            if (!rawImgPath || !rawImgPath.exists) {
                continue;
            }

            // Open base image document silently
            var doc = BT_PS.openSilent(rawImgPath);
            doc.activeLayer.name = "[BT] Original Scan";

            // Add Inpainted clean plate (placed above original scan)
            if (chkInpaint.value && inpaintImgPath && inpaintImgPath.exists) {
                try {
                    var inpaintDoc = BT_PS.openSilent(inpaintImgPath);
                    inpaintDoc.activeLayer.duplicate(doc);
                    inpaintDoc.close(SaveOptions.DONOTSAVECHANGES);
                    doc.activeLayer.name = "[BT] Clean Inpaint";
                    doc.activeLayer.visible = true;
                } catch (inpErr) {}
            }

            // Add Text Mask layer (hidden by default)
            if (chkMask.value && maskImgPath && maskImgPath.exists) {
                try {
                    var maskDoc = BT_PS.openSilent(maskImgPath);
                    maskDoc.activeLayer.duplicate(doc);
                    maskDoc.close(SaveOptions.DONOTSAVECHANGES);
                    doc.activeLayer.name = "[BT] Text Mask";
                    doc.activeLayer.visible = false;
                } catch (maskErr) {}
            }

            // Original OCR Group (Reference)
            if (chkOrig.value && pageBlocks.length > 0) {
                var origGroup = doc.layerSets.add();
                origGroup.name = "[BT] Original OCR (Reference)";
                origGroup.visible = false;

                for (var o = pageBlocks.length - 1; o >= 0; o--) {
                    var blkO = pageBlocks[o];
                    var origText = "";
                    if (blkO.text && blkO.text.join) origText = blkO.text.join("\n");
                    else if (typeof blkO.text === "string") origText = blkO.text;
                    if (!origText) continue;

                    var oLayer = origGroup.artLayers.add();
                    oLayer.kind = LayerKind.TEXT;
                    var oItem = oLayer.textItem;
                    oItem.contents = origText.replace(/\r\n/g, "\r").replace(/\n/g, "\r");
                    
                    var bRectO = blkO._bounding_rect;
                    if (!bRectO || bRectO.length < 4) {
                        if (blkO.xyxy && blkO.xyxy.length >= 4) {
                            bRectO = [blkO.xyxy[0], blkO.xyxy[1], blkO.xyxy[2] - blkO.xyxy[0], blkO.xyxy[3] - blkO.xyxy[1]];
                        } else {
                            bRectO = [50, 50, 200, 100];
                        }
                    }
                    oItem.position = [bRectO[0], bRectO[1] + 20];
                    oLayer.name = "OCR #" + (o + 1);
                }
            }

            // Translation Group
            if (chkTrans.value && pageBlocks.length > 0) {
                var transGroup = doc.layerSets.add();
                transGroup.name = "[BT] Translations";

                for (var bIdx = pageBlocks.length - 1; bIdx >= 0; bIdx--) {
                    var blk = pageBlocks[bIdx];
                    var transText = blk.translation || "";
                    if (!transText) continue;

                    var bRect = blk._bounding_rect;
                    if (!bRect || bRect.length < 4) {
                        if (blk.xyxy && blk.xyxy.length >= 4) {
                            bRect = [blk.xyxy[0], blk.xyxy[1], blk.xyxy[2] - blk.xyxy[0], blk.xyxy[3] - blk.xyxy[1]];
                        } else {
                            bRect = [50, 50, 200, 100];
                        }
                    }

                    var origW = bRect[2];
                    var origH = bRect[3];
                    var posX = bRect[0];
                    var posY = bRect[1];

                    var targetW = Math.max(origW, 40);
                    var targetH = Math.max(origH, 20);

                    // Only expand skinny 1-character vertical strips (< 70px) if Smart Bounds is enabled
                    if (chkSmartBox.value && origW < 70 && origH > origW * 1.5) {
                        targetW = Math.max(Math.round(origW * 1.8), 120);
                        targetH = Math.max(Math.round(origH * 1.05), 40);
                        posX = Math.max(0, Math.round(posX - (targetW - origW) / 2));
                    }

                    var tLayer = transGroup.artLayers.add();
                    tLayer.kind = LayerKind.TEXT;
                    var tItem = tLayer.textItem;

                    // Set text as Paragraph Box
                    if (targetW > 20 && targetH > 20) {
                        tItem.kind = TextType.PARAGRAPHTEXT;
                        tItem.width = new UnitValue(targetW, "px");
                        tItem.height = new UnitValue(targetH, "px");
                        tItem.position = [posX, posY];
                    } else {
                        tItem.kind = TextType.POINTTEXT;
                        tItem.position = [posX, posY + 20];
                    }

                    // Photoshop TextItem requires \r (Carriage Return) for newlines! \n shows up as missing glyph boxes
                    var psText = transText.replace(/\r\n/g, "\r").replace(/\n/g, "\r");
                    tItem.contents = psText;

                    // Calculate Font Size
                    var fmt = blk.fontformat || {};
                    var fontSize = fmt.font_size || blk.font_size;
                    
                    if (!fontSize || fontSize <= 0 || fontSize > 60) {
                        var charCount = transText.length;
                        if (charCount <= 8) fontSize = 24;
                        else if (charCount <= 25) fontSize = 19;
                        else if (charCount <= 60) fontSize = 16;
                        else fontSize = 14;
                    }
                    tItem.size = new UnitValue(fontSize, "px");

                    // Alignment: default to CENTER for manga dialogs
                    var align = (fmt.alignment !== undefined) ? fmt.alignment : blk._alignment;
                    if (align === 0) tItem.justification = Justification.LEFT;
                    else if (align === 2) tItem.justification = Justification.RIGHT;
                    else tItem.justification = Justification.CENTER;

                    // Fill Color
                    var frgb = fmt.frgb || [0, 0, 0];
                    var fColor = new SolidColor();
                    fColor.rgb.red = frgb[0];
                    fColor.rgb.green = frgb[1];
                    fColor.rgb.blue = frgb[2];
                    tItem.color = fColor;

                    // Auto Leading
                    try {
                        if (fmt.line_spacing && fmt.line_spacing > 5) {
                            tItem.useAutoLeading = false;
                            tItem.leading = new UnitValue(fmt.line_spacing, "px");
                        } else {
                            tItem.useAutoLeading = true;
                        }
                    } catch (ldE) {}

                    // Font Family resolution
                    var reqFont = fmt.font_family || blk.font_family;
                    var isBold = (fmt.font_weight && fmt.font_weight >= 600);
                    var isItalic = fmt.italic || false;
                    if (reqFont) {
                        var psFont = BT_PS.resolveFontPostScript(reqFont, isBold, isItalic);
                        if (psFont) {
                            try { tItem.font = psFont; } catch (fErr) {}
                        }
                    }

                    // Stroke / Outline Layer Effect (ActionManager)
                    var strokeWidth = (fmt.stroke_width !== undefined) ? fmt.stroke_width : (blk.stroke_width || 0);
                    var srgb = fmt.srgb || [255, 255, 255];
                    if (chkStroke.value) {
                        if (strokeWidth <= 0) {
                            strokeWidth = Math.max(2.5, Math.round(fontSize * 0.12));
                            srgb = [255, 255, 255];
                        }
                        BT_PS.applyStrokeFX(tLayer, strokeWidth, srgb[0], srgb[1], srgb[2], 100, "outside");
                    }

                    // Vertical Centering inside Balloon Box (matching BallonsTranslator canvas alignment)
                    if (chkVCenter.value) {
                        try {
                            if (tItem.kind === TextType.PARAGRAPHTEXT && targetH > 20) {
                                var tb = tLayer.bounds; // [left, top, right, bottom]
                                var actualTop = tb[1].as("px");
                                var actualBottom = tb[3].as("px");
                                var actualH = actualBottom - actualTop;
                                if (actualH > 0 && actualH < targetH) {
                                    var centerBoxY = posY + (targetH / 2);
                                    var centerTextY = actualTop + (actualH / 2);
                                    var shiftY = centerBoxY - centerTextY;
                                    if (Math.abs(shiftY) > 2) {
                                        tLayer.translate(0, shiftY);
                                    }
                                    // Trim excess paragraph box height so the box hugs the text neatly
                                    try {
                                        tItem.height = new UnitValue(Math.round(actualH + fontSize * 0.8), "px");
                                    } catch (hErr) {}
                                }
                            }
                        } catch (vAlignErr) {}
                    }

                    // Layer Name
                    var previewStr = transText.replace(/[\r\n]+/g, " ");
                    if (previewStr.length > 22) previewStr = previewStr.substring(0, 22) + "...";
                    tLayer.name = "#" + (bIdx + 1) + ": " + previewStr;
                }
            }

            processedCount++;
        }

        if (processedCount > 0) {
            alert(BT_I18N.t("importSuccess", { count: processedCount }), "Completed");
        }
    } finally {
        app.preferences.rulerUnits = origRuler;
        app.preferences.typeUnits = origType;
        app.displayDialogs = origDialogs;
    }
}

// Execute with graceful error and cancellation handling
try {
    runBallonTranslatorBridge();
} catch (mainErr) {
    if (!BT_PS.isUserCancelled(mainErr)) {
        alert("BallonTranslator Bridge error:\n" + mainErr.message, "Error", true);
    }
}
