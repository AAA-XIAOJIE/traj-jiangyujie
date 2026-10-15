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
    y=p("<b>公式复现：</b>OVM: dv/dt=α[V(s)-v]；FVDM: 再加 λ(v前-v)；IDM: a[1-(v/v0)^δ-(s*/s)^2]，s*=s0+s1√(v/v0)+Tv+v(v-v前)/(2√(ab))。统一净距 s=x前-x后-前车长（m），速度 m/s，加速度 m/s²。保留 IDM 2000 原式无 max 截断；OVM/FVDM 共用 Jiang 的 SI 最优速度函数。",h-63)
    y-=5
    y=p("<b>共同实验：</b>第2车道，跟车378/前车368；50.9-80.8 s 的300点校准，80.9-110.9 s 的301点验证。验证从同一真实初态自由滚动30 s；匀速/减速是反事实，仅真实前车回放计算观测误差。",y,"small")
    image_height=usable*7.2/13.2
    c.drawImage(str(Path(out)/"overview.png"),left,y-image_height-4,width=usable,height=image_height,preserveAspectRatio=True,mask="auto")
    y-=image_height+15
    c.setFillColorRGB(.08,.16,.22);c.setFont(font,10)
    c.drawString(left,y,"时间留出验证误差与减速压力测试（校准参数）");y-=17
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
    y=p(f"<b>关键发现：</b>IDM位置误差较小，FVDM速度误差较小。FVDM基线最小净距2.337 m；仅去掉速度差反馈就接触，支持该反馈的作用。独立校准后λ降至{lam:.3f} s^-1，α和C2也改变，误差改善但制动失效；不能将变化全归因于λ。跟驰拟合改善不保证制动安全。",y)
    y-=7
    y=p("<b>失败与可信度：</b>第5 s起前车以1.5 m/s²减至停；接触后终止。净距取积分节点最小值，接触时刻步内线性估计。主步长0.05 s，另核0.1/0.025 s。IDM保持正净距但需v≥0边界；395次请求仅有1次停车。校准b到4 m/s²上界，估计可能对边界敏感，不能仅凭触界断言不可识别。",y,"small")
    y-=7
    y=p("<b>能与不能：</b>OVM体现间距驱动的速度松弛；FVDM增加相对速度反馈；IDM体现速度相关期望间距与逼近制动。三者可比较本车对的速度、净距、时间间隔和TTC。时间留出的301点仍相关，单车对不能验证车流容量、车队稳定性、交通波速度、换道或跨驾驶员排名。",y,"small")
    if y<53:raise ValueError(f"Report content overlaps footer at {y:.1f}")
    c.setStrokeColorRGB(.8,.85,.88);c.line(left,40,w-left,40)
    c.setFont(font,6.8);c.setFillColorRGB(.32,.39,.43)
    c.drawString(left,29,"原文：Bando et al. (1995), PRE 51, 1035; Jiang et al. (2001), PRE 64, 017101; Treiber et al. (2000), PRE 62, 1805.")
    c.drawString(left,18,"数据：NGSIM_I80_20050413_1600-1615_SI.csv；完整公式、单位、来源与复现命令见 week05/README.md。")
    c.drawRightString(w-left,8,"1 / 1");c.showPage();c.save()
    return path
