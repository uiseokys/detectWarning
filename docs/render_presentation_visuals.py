from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "presentation_visuals"
OUT.mkdir(parents=True, exist_ok=True)

FONT_REGULAR = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\malgunbd.ttf")

INK = "#111827"
MUTED = "#64748b"
BLUE = "#2563eb"
CYAN = "#0891b2"
GREEN = "#16a34a"
YELLOW = "#d97706"
RED = "#dc2626"
PURPLE = "#7c3aed"
BG = "#f8fafc"
CARD = "#ffffff"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def canvas(width: int = 1920, height: int = 1080) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), BG)
    return image, ImageDraw.Draw(image)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=8, align="center")
    return box[2] - box[0], box[3] - box[1]


def center_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, fill=INK, spacing: int = 8) -> None:
    x, y = xy
    w, h = text_size(draw, text, fnt)
    draw.multiline_text((x - w / 2, y - h / 2), text, font=fnt, fill=fill, spacing=spacing, align="center")


def box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str = "",
    *,
    outline: str = BLUE,
    fill: str = CARD,
    title_fill: str = INK,
    body_fill: str = MUTED,
    radius: int = 22,
) -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=4)
    if body:
        center_text(draw, ((x1 + x2) // 2, y1 + 44), title, font(30, True), title_fill)
        center_text(draw, ((x1 + x2) // 2, (y1 + y2) // 2 + 28), body, font(25), body_fill)
    else:
        center_text(draw, ((x1 + x2) // 2, (y1 + y2) // 2), title, font(30, True), title_fill)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = MUTED, width: int = 6) -> None:
    draw.line([start, end], fill=color, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        points = [(ex, ey), (ex - direction * 22, ey - 14), (ex - direction * 22, ey + 14)]
    else:
        direction = 1 if ey >= sy else -1
        points = [(ex, ey), (ex - 14, ey - direction * 22), (ex + 14, ey - direction * 22)]
    draw.polygon(points, fill=color)


def title(draw: ImageDraw.ImageDraw, text: str, subtitle: str = "") -> None:
    draw.text((70, 48), text, font=font(54, True), fill=INK)
    if subtitle:
        draw.text((74, 118), subtitle, font=font(25), fill=MUTED)


def save(image: Image.Image, name: str) -> None:
    path = OUT / name
    image.save(path)
    print(path)


def render_audio_lift() -> None:
    image, d = canvas()
    title(d, "음성 사용 전/후 위험 점수 차이", "교수님께 보여주기 좋은 핵심 메시지: 영상만보다 영상+음성이 위험 판단을 더 강하게 만든다")
    box(d, (120, 250, 500, 520), "영상만", "videoOnlyScore\n62점\n주의 후보", outline=BLUE, fill="#dbeafe")
    box(d, (740, 250, 1120, 520), "음성 위험 신호", "CLOVA STT\n도움 요청·위협·통증\nAudio score 31", outline=GREEN, fill="#dcfce7")
    box(d, (1360, 250, 1800, 520), "영상+음성", "riskScore\n84점\n위험 확정", outline=RED, fill="#fee2e2")
    arrow(d, (500, 385), (740, 385), BLUE)
    arrow(d, (1120, 385), (1360, 385), GREEN)
    d.rounded_rectangle((720, 680, 1200, 860), radius=28, fill="#fff7ed", outline=YELLOW, width=5)
    center_text(d, (960, 735), "Audio gain +22", font(40, True), YELLOW)
    center_text(d, (960, 805), "음성이 들어오며 위험 점수가 상승", font(28), INK)
    arrow(d, (1520, 520), (1160, 690), RED)
    save(image, "01_audio_lift_score.png")


def render_state_machine() -> None:
    image, d = canvas()
    title(d, "위험 판단 상태 머신", "한 번 튄 점수가 아니라 반복 확인과 cooldown을 거쳐 이벤트를 보낸다")
    nodes = [
        ((120, 360, 420, 540), "정상", "Normal"),
        ((560, 360, 860, 540), "의심", "Suspicious\nwarning_min 이상"),
        ((1000, 360, 1300, 540), "확정", "Confirmed\ndanger 또는 반복 hit"),
        ((1440, 360, 1740, 540), "쿨다운", "Cooldown\n중복 알림 억제"),
    ]
    colors = [GREEN, YELLOW, RED, PURPLE]
    fills = ["#dcfce7", "#fef3c7", "#fee2e2", "#ede9fe"]
    for (xy, head, body), color, fill in zip(nodes, colors, fills):
        box(d, xy, head, body, outline=color, fill=fill)
    arrow(d, (420, 450), (560, 450), YELLOW)
    arrow(d, (860, 450), (1000, 450), RED)
    arrow(d, (1300, 450), (1440, 450), PURPLE)
    arrow(d, (1600, 540), (280, 690), MUTED)
    center_text(d, (960, 720), "클래스별 확인 조건", font(34, True), INK)
    box(d, (300, 790, 700, 940), "violence", "2회 / 8초\ncooldown 10초", outline=RED, fill="#fff1f2")
    box(d, (760, 790, 1160, 940), "fall", "1회 / 8초\ncooldown 10초", outline=BLUE, fill="#eff6ff")
    box(d, (1220, 790, 1620, 940), "loitering", "2회 / 14초\ncooldown 14초", outline=YELLOW, fill="#fffbeb")
    save(image, "02_risk_state_machine.png")


def render_fusion_matrix() -> None:
    image, d = canvas()
    title(d, "영상·음성 결합 판단 매트릭스", "음성만으로 무조건 danger가 되지 않도록 cap을 두고, 영상과 음성이 함께 맞을 때 강하게 올린다")
    x0, y0 = 360, 230
    cell_w, cell_h = 420, 190
    cols = ["음성 없음", "음성 약함", "음성 강함"]
    rows = ["영상 약함", "영상 주의", "영상 강함"]
    for i, col in enumerate(cols):
        center_text(d, (x0 + i * cell_w + cell_w // 2, y0 - 55), col, font(30, True), INK)
    for j, row in enumerate(rows):
        center_text(d, (190, y0 + j * cell_h + cell_h // 2), row, font(30, True), INK)
    cells = [
        [("관찰", "LOW/ELEVATED", GREEN, "#dcfce7"), ("의심", "audio_only_cap", YELLOW, "#fef3c7"), ("의심", "audio_only_cap 62", YELLOW, "#fef3c7")],
        [("주의 후보", "반복 확인", YELLOW, "#fef3c7"), ("주의", "weak_video_audio_cap", YELLOW, "#fffbeb"), ("위험 후보", "Audio gain 상승", RED, "#fee2e2")],
        [("영상 위험 후보", "video_only_cap", RED, "#fee2e2"), ("위험", "보강 확인", RED, "#fee2e2"), ("위험 확정", "audio+video confirmed", PURPLE, "#ede9fe")],
    ]
    for j in range(3):
        for i in range(3):
            label, sub, color, fill = cells[j][i]
            xy = (x0 + i * cell_w, y0 + j * cell_h, x0 + (i + 1) * cell_w - 20, y0 + (j + 1) * cell_h - 20)
            box(d, xy, label, sub, outline=color, fill=fill, radius=18)
    save(image, "03_fusion_matrix.png")


def render_class_guardrails() -> None:
    image, d = canvas()
    title(d, "클래스별 오탐 억제 장치", "폭력·쓰러짐·배회는 오탐 원인이 다르기 때문에 서로 다른 보정값을 사용한다")
    data = [
        ("violence", "몸싸움/빠른 움직임", "2회 반복 + 다중 사람 + 음성 확인\n영상만 cap 72/82", RED, "#fee2e2"),
        ("collapse", "앉음/숙임과 혼동", "단일 프레임 cap + temporal abnormal\n통증/도움 요청 보강", BLUE, "#dbeafe"),
        ("loitering", "기다림/정지와 혼동", "8초 지속 + 반복 hit\nwaiting pattern suppress", YELLOW, "#fef3c7"),
    ]
    for idx, (name, issue, guard, color, fill) in enumerate(data):
        x = 140 + idx * 590
        box(d, (x, 270, x + 480, 790), name, f"오탐 원인\n{issue}\n\n억제 장치\n{guard}", outline=color, fill=fill, radius=24)
    center_text(d, (960, 905), "핵심: 같은 threshold 하나로 처리하지 않고 클래스별 위험 특성에 맞게 조절", font(32, True), INK)
    save(image, "04_class_guardrails.png")


def render_privacy_boundary() -> None:
    image, d = canvas()
    title(d, "음성 개인정보 보호 경계", "STT 원문은 추론 서버 내부에서만 사용하고, 백엔드에는 위험 메타데이터만 보낸다")
    box(d, (80, 300, 420, 520), "음성 입력", "마이크 / 영상 음성", outline=GREEN, fill="#dcfce7")
    box(d, (560, 260, 980, 560), "추론 서버", "CLOVA STT\ntranscript 내부 분석\n키워드/문맥 위험 계산", outline=BLUE, fill="#dbeafe")
    box(d, (1120, 300, 1500, 520), "앱 백엔드", "audioRiskSignalDetected\naudioScore\naudioVideoGain", outline=CYAN, fill="#cffafe")
    box(d, (1580, 300, 1860, 520), "PWA", "위험 여부와 근거\n원문 없음", outline=PURPLE, fill="#ede9fe")
    arrow(d, (420, 410), (560, 410), GREEN)
    arrow(d, (980, 410), (1120, 410), CYAN)
    arrow(d, (1500, 410), (1580, 410), PURPLE)
    d.line([(640, 680), (1400, 680)], fill=RED, width=7)
    center_text(d, (1010, 725), "STT 원문 / matchedKeywords / 대화 내용은 백엔드로 전송 금지", font(33, True), RED)
    save(image, "05_audio_privacy_boundary.png")


def render_demo_storyboard() -> None:
    image, d = canvas()
    title(d, "시연 흐름 스토리보드", "카메라 입력부터 알림, 클립 확인까지 한 번에 설명하는 발표용 흐름")
    steps = [
        ("1", "카메라/테스트 영상", "프레임 + 음성 조각"),
        ("2", "추론 서버 분석", "행동 분석 + CLOVA STT"),
        ("3", "위험 점수 계산", "videoOnly / audio / fusion"),
        ("4", "백엔드 이벤트", "CCTV DB + Risk Event"),
        ("5", "PWA 알림", "SSE + Web Push"),
        ("6", "위험 클립 확인", "전 5초 + 후 10초 MP4"),
    ]
    for idx, (num, head, body) in enumerate(steps):
        x = 80 + idx * 305
        d.ellipse((x, 245, x + 72, 317), fill=BLUE)
        center_text(d, (x + 36, 281), num, font(30, True), "#ffffff")
        box(d, (x, 350, x + 250, 650), head, body, outline=BLUE, fill="#eff6ff", radius=20)
        if idx < len(steps) - 1:
            arrow(d, (x + 250, 500), (x + 305, 500), MUTED, width=5)
    center_text(d, (960, 820), "발표 멘트: 영상만 점수와 영상+음성 점수 차이를 보여준 뒤, 위험 기록에서 클립까지 확인한다.", font(31, True), INK)
    save(image, "06_demo_storyboard.png")


def main() -> None:
    render_audio_lift()
    render_state_machine()
    render_fusion_matrix()
    render_class_guardrails()
    render_privacy_boundary()
    render_demo_storyboard()


if __name__ == "__main__":
    main()
