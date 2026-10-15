"""One-page course result from the actual three-model comparisons."""
from pathlib import Path
import os

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph


def one_page(out, metrics, parameters):
    font="STSong-Light"
    for candidate in [os.environ.get("TRAJ_CJK_FONT",""),"C:/Windows/Fonts/simsun.ttc",
                      "/usr/share/fonts/truetype/arphic/uming.ttc"]:
        if candidate and Path(candidate).is_file():
            pdfmetrics.registerFont(TTFont("Week05CJK",candidate,subfontIndex=0))
            pdfmetrics.registerFontFamily("Week05CJK",normal="Week05CJK",bold="Week05CJK",italic="Week05CJK",boldItalic="Week05CJK")
            font="Week05CJK";break
    else:
        pdfmetrics.registerFont(UnicodeCIDFont(font))
    path=Path(out)/"one_page.pdf"
    c=canvas.Canvas(str(path),pagesize=A4,pageCompression=1,invariant=1)
    c.setTitle("Week 05 - OVM FVDM IDM reproduction and comparison");c.setAuthor("Jiang Yujie")
    w,h=A4;left=30;usable=w-60
    c.setFillColorRGB(.08,.16,.22);c.setFont(font,17)
    c.drawString(left,h-35,"第5周：三种跟驰模型的复现与比较")
    c.setFont(font,9);c.drawString(left,h-51,"OVM / FVDM / IDM | NGSIM I-80, 2005-04-13 16:00-16:15 | 姜昱杰")
    styles={k:ParagraphStyle(k,fontName=font,fontSize=size,leading=leading,textColor="#243743")
            for k,size,leading in [("body",9,12.5),("small",7.5,10)]}
    def p(text,y,style="body"):
        paragraph=Paragraph(text,styles[style]);_,height=paragraph.wrap(usable,1000)
        paragraph.drawOn(c,left,y-height);return y-height
    y=p("<b>公式复现：</b>OVM: dv/dt=κ[V(s)-v]；FVDM: 再加 λ(v前-v)；IDM: a[1-(v/v0)^δ-(s*/s)^2]，s*=s0+s1√(v/v0)+Tv+v(v-v前)/(2√(ab))。统一净距 s=x前-x后-前车长（m），速度 m/s，加速度 m/s²。保留 IDM 2000 原式无 max 截断；OVM/FVDM 共用 Jiang 的 SI 最优速度函数。",h-63)
    y-=5
    y=p("<b>共同实验：</b>第2车道，跟车378/前车368；50.9-80.8 s 的300点校准，80.9-110.9 s 的301点验证。验证从同一真实初态自由滚动30 s；匀速/减速是反事实，仅真实前车回放计算观测误差。",y,"small")
    image_height=usable*7.2/13.2
    c.drawImage(str(Path(out)/"overview.png"),left,y-image_height-4,width=usable,height=image_height,preserveAspectRatio=True,mask="auto")
    y-=image_height+15
    c.setFillColorRGB(.08,.16,.22);c.setFont(font,10)
    c.drawString(left,y,"独立验证误差与减速压力测试（校准参数）");y-=17
    columns=[left,left+55,left+155,left+270,left+385]
    c.setFillColorRGB(.92,.95,.97);c.rect(left-3,y-5,usable+6,18,fill=1,stroke=0)
    c.setFillColorRGB(.08,.16,.22);c.setFont(font,8.3)
    for x,label in zip(columns,["模型","位置RMSE (m)","速度RMSE (m/s)","减速最小净距 (m)","碰撞/数值边界"]):c.drawString(x,y,label)
    calibrated=metrics[metrics.parameter_set=="calibrated"]
    for model in ["OVM","FVDM","IDM"]:
        row=calibrated[(calibrated.model==model)&(calibrated.scenario=="observed")].iloc[0]
        brake=calibrated[(calibrated.model==model)&(calibrated.scenario=="braking")].iloc[0]
        label=f"{brake.collision_time_s:.2f} s 接触" if brake.status=="collision" else "未碰撞；需v≥0边界"
        y-=17
        for x,label in zip(columns,[model,f"{row.position_rmse_m:.3f}",f"{row.speed_rmse_m_s:.3f}",f"{brake.minimum_gap_m:.3f}",label]):c.drawString(x,y,label)
    y-=15
    lam=parameters["calibrated"]["FVDM"]["lambda"]
    y=p(f"<b>关键发现：</b>IDM位置误差较小，FVDM速度误差较小；不存在对所有指标都最好的模型。FVDM的λ由基线0.5降至{lam:.3f} s^-1，拟合改善但制动阻尼变弱：原基线减速最小净距2.337 m、未碰撞，校准后发生接触。仅最小化普通跟驰误差不能保证紧急制动安全。",y)
    y-=7
    y=p("<b>失败与可信度：</b>减速从第5 s开始，前车以1.5 m/s²减至停；碰撞后终止并保留缺失，不修复净距。主步长0.05 s，另核0.1/0.025 s。IDM保持正净距，但需非负速度边界；395次请求包含静止期间重复负加速度请求，并非395次停车。IDM校准b到4 m/s²上界，提示参数识别有限。",y,"small")
    y-=7
    y=p("<b>能与不能：</b>OVM体现间距驱动的速度松弛；FVDM增加相对速度反馈；IDM体现速度相关期望间距与逼近制动。三者可比较本车对的速度、净距、时间间隔和TTC；单车对不能验证车流容量、车队稳定性、交通波速度、换道或所有驾驶人的表现。数值边界与模型原式已分别披露。",y,"small")
    if y<53:raise ValueError(f"Report content overlaps footer at {y:.1f}")
    c.setStrokeColorRGB(.8,.85,.88);c.line(left,40,w-left,40)
    c.setFont(font,6.8);c.setFillColorRGB(.32,.39,.43)
    c.drawString(left,29,"原文：Bando et al. (1995), PRE 51, 1035; Jiang et al. (2001), PRE 64, 017101; Treiber et al. (2000), PRE 62, 1805.")
    c.drawString(left,18,"数据：NGSIM_I80_20050413_1600-1615_SI.csv；完整公式、单位、来源与复现命令见 week05/README.md。")
    c.drawRightString(w-left,8,"1 / 1");c.showPage();c.save()
    return path
