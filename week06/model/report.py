"""One-page Chinese course handout generated from the actual metrics."""
import os
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph


def one_page(out,metrics):
    font="STSong-Light"
    for path in [os.environ.get("TRAJ_CJK_FONT",""),"C:/Windows/Fonts/simsun.ttc","/usr/share/fonts/truetype/arphic/uming.ttc"]:
        if path and Path(path).is_file():
            font="Week06CJK";pdfmetrics.registerFont(TTFont(font,path,subfontIndex=0));break
    else:pdfmetrics.registerFont(UnicodeCIDFont(font))
    pdfmetrics.registerFontFamily(font,normal=font,bold=font,italic=font,boldItalic=font)
    path=out/"one_page.pdf";c=canvas.Canvas(str(path),pagesize=A4,invariant=1,pageCompression=1)
    w,h=A4;left=30;usable=w-60
    c.setTitle("Week06 - Unified pedestrian and vehicle modeling");c.setAuthor("Jiang Yujie")
    c.setFont(font,17);c.setFillColorRGB(.09,.19,.25)
    c.drawString(left,h-34,"第6周：同一交互模型，能否统一人和车？")
    styles={k:ParagraphStyle(k,fontName=font,fontSize=size,leading=lead,textColor="#263D48")
        for k,size,lead in [("body",9,12),("small",8,10.5)]}
    def para(text,y,style="body"):
        p=Paragraph(text,styles[style]);_,height=p.wrap(usable,1000);p.drawOn(c,left,y-height);return y-height
    y=para("<b>统一范围：</b>每组实验中，人和车调用同一个 SFM、IDM-heading 或 ORCA 交互核心；尺寸、期望速度和可行动作不同。窄路车辆直行，交替圆环另设有限转向改进模型，车辆不能瞬间横移。",h-49)
    y-=6
    y=para("<b>设置：</b>6人迎面遇1车，行人半径0.25 m、期望速度1.4 m/s；汽车4.4×1.8 m、3 m/s。只改变路宽4.2/3.2/2.9/2.6 m，仿真40 s、步长0.04 s。汽车居中时，单侧并行最低路宽2.8 m。另比较真实64人圆环，以及半径30 m、32人+32车交替的合成圆环；车辆有限转向作为改良模型。",y,"small")
    image_h=usable*6.8/12
    c.drawImage(str(out/"failure_and_repair.png"),left,y-image_h-3,width=usable,height=image_h,mask="auto")
    y-=image_h+16
    y=para("<b>3.2 m 窄路结果：到达、几何有效、动力学有效必须分开判断</b>",y)
    columns=[left,left+65,left+132,left+225,left+340,left+435]
    y-=17;c.setFillColorRGB(.91,.95,.96);c.rect(left-3,y-5,usable+6,17,fill=1,stroke=0)
    c.setFillColorRGB(.09,.19,.25);c.setFont(font,8)
    for x,text in zip(columns,["模型","版本","退出主体数","最小净距(m)","峰值加速度","综合有效"]):c.drawString(x,y,text)
    for model in ["sfm","idm","orca"]:
        for variant in ["base","guided"]:
            r=metrics[(metrics.scene=="narrow")&(metrics.width_m==3.2)&(metrics.model==model)&(metrics.variant==variant)].iloc[0]
            y-=14
            for x,text in zip(columns,[model.upper(),"基础" if variant=="base" else "改良",f"{int(r.arrived)}/7",f"{r.minimum_body_gap_m:.3f}",f"{r.max_realized_acceleration_m_s2:.2f} m/s²","是" if r.physically_feasible_complete else "否"]):c.drawString(x,y,text)
    y-=14
    r=metrics[(metrics.scene=="circle")&(metrics.model=="orca")&(metrics.variant=="guided")].iloc[0]
    y=para(f"<b>圆环实测对照：</b>保留原始64人起点、对跖目标和前2 s初始化参数。ORCA改良到达{int(r.arrived)}/64，2 s后位置ADE={r.ade_after_calibration_m:.2f} m，最小假设身体净距{r.minimum_body_gap_m:.3f} m；仍有重叠，不能宣称严格安全。SFM会压缩重叠，IDM-heading容易停滞。真实记录只有行人；混合圆环没有实测误差指标。",y,"small")
    y-=7
    mixed=metrics[(metrics.scene=="circle_mixed")&(metrics.variant=="steering")]
    details="；".join(f"{r.model.upper()}到达{int(r.pedestrians_arrived)}人/{int(r.cars_arrived)}车，包络净距{r.minimum_body_gap_m:.2f} m" for r in mixed.itertuples())
    y=para("<b>交替人车圆环：</b>"+details+"。转向受轴距2.7 m、转角30°、角速度0.35 rad/s、横向加速度2 m/s²限制；车辆使用外接圆碰撞包络，重叠仅说明包络冲突，不能直接认定实际车身相撞。",y,"small")
    y-=7
    y=para("<b>主要认识：</b>3.2 m社会力基础版存在重叠/越界，侧向通路规则可修复；2.9 m有空间却仍可能僵持，是规则不足；2.6 m不能并行通过，等待符合几何限制。几何安全干预可能产生过大加速度，不能把“没有重叠”直接说成“运动合理”。车辆的形状与动作约束，是统一建模不可省略的一部分。",y,"small")
    y-=7
    y=para("<b>核查与局限：</b>单步扫掠几何检测、双车矩形尺寸、出口移除、同宽度初态、步长减半和规则消融均可复算。圆环为同记录机制比较，非独立预测验证；模型没有经过多场景标定。",y,"small")
    if y<49:raise ValueError(f"Report overflows at {y}")
    c.setStrokeColorRGB(.78,.84,.87);c.line(left,38,w-left,38)
    c.setFont(font,6.5)
    c.drawString(left,26,"来源：Helbing & Molnar (1995); Treiber et al. (2000); van den Berg et al. (2011); Xiao et al. (2018).")
    c.drawString(left,16,"代码与完整参数：week06/README.md；AI辅助：Codex检索、实现与核查。数值、违规与失败均来自脚本运行。")
    c.drawRightString(w-left,7,"1 / 1");c.showPage();c.save();return path
