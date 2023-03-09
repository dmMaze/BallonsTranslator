import re
import os.path as osp
from pathlib import Path

def set_svgcolor(svgpath, savename, color):
    fillcolor = "fill=\"" + color + "\""
    if savename is not None:
        savepath = osp.join(osp.dirname(svgpath), savename + ".svg")
    else:
        savepath = svgpath
    subs = None
    with open(svgpath, "r", encoding="utf-8") as f:
        fread = f.read()
        if re.findall(r'fill=\"(.*?)\"', fread):
            subs = re.sub(r'fill=\"(.*?)\"', lambda matchedobj: fillcolor, fread)
        else:
            subs = re.sub(r'p-id=\"(.*?)\"', lambda matchedobj: matchedobj.group(0) + ' ' + fillcolor, fread)
    with open(savepath, "w", encoding="utf-8") as f:
        f.write(subs)

svgtemplate = r'<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"><svg t="1613973513102" class="icon" viewBox="VIEWBOX" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="1676" width="128" height="128" xmlns:xlink="http://www.w3.org/1999/xlink"><defs><style type="text/css"></style></defs>PATH</svg>'

def minify_svg(svgpath):
    '''
    convert png to svg
    https://png-to-svg.com/
    then use following script to minify svg
    '''

    with open(svgpath, "r", encoding="utf-8") as f:
        fread = f.read()
        p = re.findall(r'<path d=(.*?)</path>', fread)[0]
        p = r'<path d=' + p + r'</path>'
        v = re.findall(r'viewBox=\"(.*?)\"', fread)[0]
        svg = svgtemplate.replace("PATH", p)
        svg = svg.replace("VIEWBOX", v)
        with open(svgpath, "w", encoding="utf-8") as f:
            f.write(svg)


if __name__ == '__main__':

    eva_dark = "#697187"
    eva_light = "#b3b6bf"
    fontcolor = "#939395"
    white = "#ffffff"
    smokewhite = "#f5f5f5"


    svgpath =  r'data\icons\cursor_rotate_0.svg'

    # savename = r'titlebar_close.svg'
    # colored_activate_savename = r'imgtrans_activate'
    colored_savename = r'drawingtools_inpaint_activate'
    minify_svg(svgpath)

    set_svgcolor(svgpath, colored_savename, eva_light)