"""One-page Chinese course handout, with actual results embedded."""
from pathlib import Path
import os

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle


def one_page(out, summary, audit):
    candidates=[os.environ.get("TRAJ_CJK_FONT", ""), "C:/Windows/Fonts/simsun.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc"]
    font="STSong-Light"
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            pdfmetrics.registerFont(TTFont("CourseCJK", candidate, subfontIndex=0))
            pdfmetrics.registerFontFamily("CourseCJK", normal="CourseCJK", bold="CourseCJK", italic="CourseCJK", boldItalic="CourseCJK")
            font="CourseCJK"
            break
    else:
        pdfmetrics.registerFont(UnicodeCIDFont(font))
    output=Path(out)/"one_page.pdf"
    c=canvas.Canvas(str(output),pagesize=A4,pageCompression=1,invariant=1)
    c.setTitle("Week 04 - NGSIM I-80 fundamental diagram comparison")
    c.setAuthor("Jiang Yujie")
    width,height=A4;left=32;usable=width-64
    c.setFillColorRGB(.08,.16,.22);c.setFont(font,17)
    c.drawString(left,height-36,"第4周：怎样用轨迹较好地表征交通流？")
    c.setFont(font,9)
    c.drawString(left,height-54,"NGSIM I-80 / 2005-04-13 16:00-16:15 PDT / 姜昱杰")
    styles={"body":ParagraphStyle("body",fontName=font,fontSize=9,leading=13,textColor="#243743"),
            "small":ParagraphStyle("small",fontName=font,fontSize=7.5,leading=10,textColor="#53636B")}
    def paragraph(text,y,style="body"):
        p=Paragraph(text,styles[style]);_,h=p.wrap(usable,1000);p.drawOn(c,left,y-h);return y-h
    y=paragraph("<b>统一范围：</b>东向第2车道，纵向150-250 m（100 m）；主分析0-840 s。T=30 s，各方法28个共同有效窗口。输入为课程SI版；车辆前端中心作为轨迹参考点。",height-68)
    y=paragraph("源文件：NGSIM_I80_20050413_1600-1615_SI.csv",y-2,"small")
    img_h=usable*7.7/11.4
    c.drawImage(str(Path(out)/"methods.png"),left,y-img_h-5,width=usable,height=img_h,preserveAspectRatio=True,mask="auto")
    y-=img_h+17
    names={"edie":"时空积分 Edie","detector":"虚拟断面","voronoi":"一维 Voronoi","kernel":"三角核平滑"}
    c.setFillColorRGB(.08,.16,.22);c.setFont(font,10)
    c.drawString(left,y,"同一时段的平均状态（单车道）")
    y-=17
    columns=[left,left+155,left+270,left+400]
    c.setFillColorRGB(.92,.95,.97);c.rect(left-3,y-5,usable+6,18,fill=1,stroke=0)
    c.setFillColorRGB(.08,.16,.22);c.setFont(font,9)
    for x,s in zip(columns,["方法","密度 k (veh/km)","流量 q (veh/h)","速度 q/k (km/h)"]):c.drawString(x,y,s)
    for method in names:
        y-=17;r=summary[(summary.method==method)&(summary.window_s==30)].iloc[0]
        vals=[names[method],f"{r.mean_k_veh_km:.2f}",f"{r.mean_q_veh_h:.1f}",f"{r.mean_v_km_h:.2f}"]
        for x,s in zip(columns,vals):c.drawString(x,y,s)
    y-=17
    e=summary[summary.method=="edie"].set_index("window_s")
    reduction=(1-e.loc[60].q_sd/e.loc[10].q_sd)*100
    y=paragraph(f"<b>窗口不是新方法：</b>Edie的10/30/60 s窗口分别得到84/28/14个点，流量标准差为{e.loc[10].q_sd:.0f}/{e.loc[30].q_sd:.0f}/{e.loc[60].q_sd:.0f} veh/h；10 s变为60 s后降低{reduction:.1f}%。平均流量保持{e.loc[30].mean_q_veh_h:.1f} veh/h，变化来自状态混合与分辨率。",y)
    y-=8
    y=paragraph("<b>解释与心得：</b>断面计数与100 m路段积分的测量对象不同，短窗下的散点差异更明显。Voronoi与核平滑能减轻车辆跨边界引起的跳变，但依赖外部邻车或核宽。应以边界裁剪严格、流速密口径一致的Edie作为路段统计参照，保留散点并结合时空速度图，再做时间窗、空间尺度与窗口起点的灵敏度分析；不能以点云更紧或拟合更漂亮判定方法更正确。",y)
    y-=8
    y=paragraph("<b>质量与限制：</b>全源1,226,650条记录无缺失、无重复车辆-帧键。末段883.6-899.9 s无法构造完整Voronoi胞元，因此主比较统一取前840 s，保证三种时间窗完整；输入保留全部原始支持轨迹。原始位置仍含瞬时速度尖峰，已审计并保留。本次曲线仅描述所观测状态，不外推容量或堵塞密度。",y,"small")
    if y < 48:
        raise ValueError(f"One-page report would overlap footer: y={y}")
    c.setStrokeColorRGB(.8,.85,.88);c.line(left,36,width-left,36)
    c.setFont(font,7.2);c.setFillColorRGB(.32,.39,.43)
    c.drawString(left,25,"数据：USDOT NGSIM / 方法：Edie、Steffen & Seyfried / 公式、来源和复现命令见 week04/README.md")
    c.drawRightString(width-left,14,"1 / 1")
    c.showPage();c.save()
    return output
