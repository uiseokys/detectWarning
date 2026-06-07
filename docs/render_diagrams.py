from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

FONT_REGULAR = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\malgunbd.ttf")

SCALE = 2
W, H = 1920, 1080
BG = "#f7fafc"
INK = "#111827"
MUTED = "#475569"

THEMES = {
    "blue": ("#dbeafe", "#2563eb"),
    "cyan": ("#ecfeff", "#0891b2"),
    "green": ("#dcfce7", "#16a34a"),
    "mint": ("#d1fae5", "#059669"),
    "red": ("#fee2e2", "#dc2626"),
    "amber": ("#fef3c7", "#d97706"),
    "orange": ("#ffedd5", "#ea580c"),
    "violet": ("#ede9fe", "#7c3aed"),
    "slate": ("#f8fafc", "#64748b"),
    "white": ("#ffffff", "#334155"),
}


def s(value: float) -> int:
    return int(round(value * SCALE))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), s(size))


def new_canvas(title: str, width: int = W, height: int = H) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (s(width), s(height)), BG)
    draw = ImageDraw.Draw(image)
    draw.text((s(52), s(48)), title, font=font(42, True), fill=INK)
    return image, draw


def multiline_size(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=s(8), align="center")
    return box[2] - box[0], box[3] - box[1]


def centered_text(
    draw: ImageDraw.ImageDraw,
    xywh: tuple[float, float, float, float],
    text: str,
    size: int = 25,
    bold: bool = True,
    fill: str = INK,
) -> None:
    x, y, w, h = [s(v) for v in xywh]
    text_font = font(size, bold)
    tw, th = multiline_size(draw, text, text_font)
    draw.multiline_text(
        (x + w / 2 - tw / 2, y + h / 2 - th / 2),
        text,
        font=text_font,
        fill=fill,
        spacing=s(8),
        align="center",
    )


def box(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    theme: str = "white",
    size: int = 25,
    radius: int = 18,
) -> None:
    fill, outline = THEMES[theme]
    draw.rounded_rectangle(
        (s(x), s(y), s(x + w), s(y + h)),
        radius=s(radius),
        fill=fill,
        outline=outline,
        width=s(4),
    )
    centered_text(draw, (x, y, w, h), text, size=size, bold=True)


def label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    size: int = 22,
    fill: str = MUTED,
) -> None:
    draw.text((s(x), s(y)), text, font=font(size, True), fill=fill, anchor="mm")


def arrow(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#334155",
    text: str | None = None,
    width: int = 5,
) -> None:
    x1i, y1i, x2i, y2i = map(s, (x1, y1, x2, y2))
    draw.line((x1i, y1i, x2i, y2i), fill=color, width=s(width))
    angle = math.atan2(y2i - y1i, x2i - x1i)
    head = s(20)
    wing = s(11)
    points = [
        (x2i, y2i),
        (
            int(x2i - head * math.cos(angle) + wing * math.sin(angle)),
            int(y2i - head * math.sin(angle) - wing * math.cos(angle)),
        ),
        (
            int(x2i - head * math.cos(angle) - wing * math.sin(angle)),
            int(y2i - head * math.sin(angle) + wing * math.cos(angle)),
        ),
    ]
    draw.polygon(points, fill=color)
    if text:
        draw.text(
            ((x1i + x2i) // 2, (y1i + y2i) // 2 - s(20)),
            text,
            font=font(20, True),
            fill=color,
            anchor="mm",
        )


def poly_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: str = "#334155",
    text: str | None = None,
    width: int = 5,
) -> None:
    scaled = [(s(x), s(y)) for x, y in points]
    draw.line(scaled, fill=color, width=s(width), joint="curve")
    x1, y1 = scaled[-2]
    x2, y2 = scaled[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    head = s(20)
    wing = s(11)
    draw.polygon(
        [
            (x2, y2),
            (
                int(x2 - head * math.cos(angle) + wing * math.sin(angle)),
                int(y2 - head * math.sin(angle) - wing * math.cos(angle)),
            ),
            (
                int(x2 - head * math.cos(angle) - wing * math.sin(angle)),
                int(y2 - head * math.sin(angle) + wing * math.cos(angle)),
            ),
        ],
        fill=color,
    )
    if text:
        mx, my = scaled[len(scaled) // 2]
        draw.text((mx, my - s(20)), text, font=font(20, True), fill=color, anchor="mm")


def save(image: Image.Image, filename: str) -> None:
    path = OUT / filename
    image.save(path, "PNG")
    print(f"{path} {image.size[0]}x{image.size[1]}")


def render_architecture() -> None:
    image, d = new_canvas("detectWarning 전체 시스템 아키텍처")
    box(d, 70, 190, 340, 120, "카메라 업로더\nMac / Windows\n웹캠·아이폰 카메라", "blue", 24)
    box(d, 70, 435, 340, 110, "영상 테스트\n대시보드 파일 업로드", "violet", 24)
    box(d, 70, 685, 340, 110, "음성 입력\n마이크 / 영상 음성", "mint", 24)
    box(d, 615, 405, 390, 155, "추론 서버 :8001\nFastAPI 대시보드\n세션·프레임·STT 관리", "white", 27)
    box(d, 1185, 155, 360, 100, "앱 백엔드 :8000\nCCTV DB / 이벤트 / PWA", "cyan", 24)
    box(d, 1580, 155, 270, 80, "PWA / iPhone\n사용자 화면", "slate", 23)
    box(d, 1240, 315, 390, 100, "YOLO / Pose\n사람 탐지·관절점", "blue", 25)
    box(d, 1240, 455, 390, 100, "실시간 행동 분석\nviolence / collapse / loitering", "amber", 22)
    box(d, 1240, 595, 390, 100, "CLOVA STT\n음성 원문은 내부 전용", "green", 23)
    box(d, 1240, 760, 390, 100, "RiskAnalyzer\n영상+음성 결합 점수", "red", 23)
    box(d, 615, 760, 390, 100, "위험 클립 서비스\n판단 전 5초 + 후 10초 MP4", "violet", 22)
    arrow(d, 410, 250, 615, 455, "#2563eb", "JPEG 프레임")
    arrow(d, 410, 490, 615, 490, "#7c3aed", "파일 루프")
    arrow(d, 410, 740, 615, 535, "#059669", "음성 조각")
    arrow(d, 1005, 450, 1240, 365, "#2563eb")
    arrow(d, 1005, 485, 1240, 505, "#d97706")
    arrow(d, 1005, 525, 1240, 645, "#16a34a")
    arrow(d, 1435, 415, 1435, 455, "#cbd5e1", width=4)
    arrow(d, 1435, 555, 1435, 595, "#cbd5e1", width=4)
    arrow(d, 1435, 695, 1435, 760, "#cbd5e1", width=4)
    arrow(d, 1240, 810, 1005, 810, "#dc2626", "위험 클립")
    arrow(d, 810, 405, 1270, 255, "#0891b2", "상태·이벤트")
    arrow(d, 1545, 195, 1580, 195, "#0891b2", "WebRTC / SSE")
    save(image, "01_system_architecture.png")


def render_sequence() -> None:
    image, d = new_canvas("실시간 감지 시퀀스")
    actors = [("업로더", 180), ("추론 서버", 500), ("Detector/Action", 820), ("CLOVA STT", 1140), ("RiskAnalyzer", 1460), ("백엔드/PWA", 1720)]
    for name, x in actors:
        box(d, x - 100, 150, 200, 72, name, "white", 22)
        d.line((s(x), s(222), s(x), s(900)), fill="#cbd5e1", width=s(3))
    steps = [
        (180, 500, 300, "POST /analyze/frame"),
        (500, 820, 390, "사람·자세·행동 분석"),
        (820, 500, 480, "action result"),
        (180, 500, 570, "POST /analyze/audio"),
        (500, 1140, 650, "CLOVA 요청"),
        (1140, 500, 730, "음성 위험 메타데이터"),
        (500, 1460, 810, "영상+음성 결합"),
        (1460, 1720, 890, "/inference/state/events"),
    ]
    for x1, x2, y, text in steps:
        arrow(d, x1, y, x2, y, "#334155", text)
    save(image, "02_realtime_sequence.png")


def render_risk_scoring() -> None:
    image, d = new_canvas("위험 점수 산정 로직")
    box(d, 100, 210, 320, 110, "분석 프레임\nPose / RGB / I3D", "blue", 25)
    box(d, 100, 590, 320, 110, "음성 조각\nCLOVA STT / 소리 특징", "green", 24)
    box(d, 600, 210, 300, 95, "Video-only score", "white", 24)
    box(d, 600, 405, 300, 95, "시간 안정화\nstreak / cooldown", "amber", 24)
    box(d, 600, 590, 300, 105, "Audio score\n발화·비명·충격음", "white", 23)
    box(d, 1110, 395, 340, 120, "Fusion score\n영상+음성 결합", "red", 27)
    box(d, 1110, 640, 340, 100, "오탐 완화\n클래스별 threshold", "violet", 24)
    for x, text, theme in [(300, "정상", "green"), (620, "의심", "amber"), (940, "주의", "orange"), (1260, "위험", "red")]:
        box(d, x, 850, 210, 80, text, theme, 25)
    arrow(d, 420, 265, 600, 260, "#2563eb")
    arrow(d, 420, 645, 600, 640, "#16a34a")
    arrow(d, 750, 305, 750, 405, "#d97706")
    arrow(d, 900, 452, 1110, 455, "#dc2626")
    arrow(d, 900, 640, 1110, 480, "#16a34a")
    arrow(d, 1280, 515, 1280, 640, "#7c3aed")
    for x in [405, 725, 1045, 1365]:
        arrow(d, 1280, 740, x, 850, "#64748b", width=4)
    label(d, 960, 1000, "audioVideoGain = 최종 점수 - 영상만 점수", 24, "#dc2626")
    save(image, "03_risk_scoring_logic.png")


def render_audio_privacy() -> None:
    image, d = new_canvas("CLOVA 음성 인식 및 개인정보 흐름")
    nodes = [
        (80, 360, 260, 100, "마이크/영상 음성", "green"),
        (410, 360, 260, 100, "음성 보정\ngain / RMS / peak", "white"),
        (740, 360, 280, 100, "업로드 gate\n짧음·무음·요청 제한", "amber"),
        (1090, 360, 230, 100, "CLOVA CSR", "blue"),
        (1390, 360, 310, 100, "Transcript\n추론 서버 내부 전용", "slate"),
        (620, 660, 340, 110, "위험 메타데이터\naudioRiskSignalDetected\naudioScore", "red"),
        (1110, 660, 340, 110, "앱 백엔드\n원문 저장·전송 없음", "cyan"),
    ]
    for x, y, w, h, text, theme in nodes:
        box(d, x, y, w, h, text, theme, 23)
    for start, end in [((340, 410), (410, 410)), ((670, 410), (740, 410)), ((1020, 410), (1090, 410)), ((1320, 410), (1390, 410)), ((1540, 460), (790, 660)), ((960, 715), (1110, 715))]:
        arrow(d, *start, *end, "#334155")
    label(d, 1290, 570, "STT 원문 / matchedKeywords는 백엔드 전송 금지", 26, "#dc2626")
    save(image, "04_audio_privacy_flow.png")


def render_pairing() -> None:
    image, d = new_canvas("카메라별 CCTV 코드 연결")
    box(d, 140, 300, 360, 120, "추론 서버\nclient_id별 코드 생성/조회", "blue", 25)
    box(d, 710, 300, 370, 120, "앱 백엔드\nPOST /inference/cctvs\ncode 기준 upsert", "cyan", 23)
    box(d, 1320, 300, 300, 120, "PWA 사용자\nCCTV 코드 입력", "green", 25)
    box(d, 710, 620, 370, 120, "user_cctvs\n사용자-CCTV 연결", "white", 25)
    box(d, 1320, 620, 300, 120, "동등 권한\n같은 코드 사용자 공유", "violet", 24)
    arrow(d, 500, 360, 710, 360, "#2563eb", "code/name/status")
    arrow(d, 1320, 360, 1080, 360, "#16a34a", "코드 입력")
    arrow(d, 895, 420, 895, 620, "#0891b2")
    arrow(d, 1080, 680, 1320, 680, "#7c3aed")
    save(image, "05_cctv_pairing_flow.png")


def render_webrtc() -> None:
    image, d = new_canvas("WebRTC direct-first 송출 구조")
    actors = [("PWA", 220), ("앱 백엔드", 650), ("추론 서버", 1100), ("LatestJpegVideoTrack", 1540)]
    for name, x in actors:
        box(d, x - 130, 150, 260, 78, name, "white", 22)
        d.line((s(x), s(228), s(x), s(850)), fill="#cbd5e1", width=s(3))
    steps = [
        (220, 650, 320, "offer"),
        (650, 1100, 430, "offer direct-first"),
        (1100, 1540, 540, "video track 생성"),
        (1540, 1100, 640, "분석 프레임"),
        (1100, 650, 740, "answer"),
        (650, 220, 840, "answer + upstream"),
    ]
    for x1, x2, y, text in steps:
        arrow(d, x1, y, x2, y, "#2563eb" if "offer" in text or "answer" in text else "#16a34a", text)
    box(d, 440, 930, 1040, 70, "백엔드는 JPEG를 WebRTC로 변환하지 않고 offer/answer를 중계한다", "amber", 25)
    save(image, "06_webrtc_direct_first.png")


def render_clip_pipeline() -> None:
    image, d = new_canvas("위험 구간 클립 저장 파이프라인")
    for x, text, theme in [
        (120, "분석 프레임\n링버퍼 20~30초", "blue"),
        (520, "위험 이벤트\n판단 시점", "red"),
        (920, "프레임 선택\n전 5초 + 후 10초", "amber"),
        (1320, "MP4 생성\nRange 요청 지원", "violet"),
    ]:
        box(d, x, 390, 300, 125, text, theme, 24)
    for x1, x2 in [(420, 520), (820, 920), (1220, 1320)]:
        arrow(d, x1, 452, x2, 452, "#334155")
    box(d, 520, 660, 850, 110, "clipUrl을 백엔드 이벤트 payload에 포함\nSTT 원문은 저장하거나 전송하지 않음", "white", 25)
    arrow(d, 1470, 515, 1120, 660, "#7c3aed")
    save(image, "07_event_clip_pipeline.png")


def render_erd() -> None:
    width, height = 2400, 1500
    image = Image.new("RGB", (s(width), s(height)), BG)
    d = ImageDraw.Draw(image)
    d.text((s(60), s(50)), "앱 백엔드 ERD", font=font(48, True), fill=INK)

    def table(x: int, y: int, w: int, title: str, fields: list[str]) -> tuple[int, int, int, int]:
        h = 120 + len(fields) * 48
        fill, outline = THEMES["white"]
        d.rounded_rectangle((s(x), s(y), s(x + w), s(y + h)), radius=s(18), fill=fill, outline=outline, width=s(4))
        d.text((s(x + w / 2), s(y + 35)), title, font=font(25, True), fill="#2563eb", anchor="mm")
        d.line((s(x), s(y + 68), s(x + w), s(y + 68)), fill=outline, width=s(3))
        for i, field in enumerate(fields):
            d.text((s(x + 24), s(y + 98 + i * 48)), field, font=font(20, False), fill=INK)
        return x, y, w, h

    tables = {
        "USERS": table(80, 170, 390, "USERS", ["pk PK", "email", "password_hash", "name", "created_at"]),
        "SESSIONS": table(80, 570, 390, "SESSIONS", ["token PK", "user_pk FK", "created_at", "expires_at"]),
        "USER_PROFILES": table(80, 980, 390, "USER_PROFILES", ["user_pk PK/FK", "display_name", "phone", "organization"]),
        "PUSH_SUBS": table(620, 170, 440, "PUSH_SUBSCRIPTIONS", ["id PK", "user_pk FK", "endpoint", "p256dh/auth"]),
        "USER_CCTVS": table(620, 620, 440, "USER_CCTVS", ["user_pk PK/FK", "cctv_id PK/FK", "created_at"]),
        "CCTVS": table(1210, 360, 470, "CCTVS", ["id PK", "code UK", "name/location/status", "latest_risk_score", "inference_client_id", "thresholds"]),
        "RISK_EVENTS": table(1210, 900, 470, "RISK_EVENTS", ["id PK", "cctv_id FK", "risk_level / class", "video/audio/fusion score", "clip_url", "incident_key"]),
        "SCORE": table(1820, 320, 400, "CCTV_SCORE_SAMPLES", ["id PK", "cctv_id FK", "score", "observed_at", "source"]),
        "HEALTH": table(1820, 830, 400, "CCTV_HEALTH_EVENTS", ["id PK", "cctv_id FK", "status", "reason", "observed_at"]),
    }

    def side(rect: tuple[int, int, int, int], name: str) -> tuple[int, int]:
        x, y, w, h = rect
        return {"r": (x + w, y + h // 2), "l": (x, y + h // 2), "b": (x + w // 2, y + h), "t": (x + w // 2, y)}[name]

    def connector(a: str, b: str, relation: str, sa: str, sb: str, color: str = "#64748b") -> None:
        arrow(d, *side(tables[a], sa), *side(tables[b], sb), color, relation, width=4)

    connector("USERS", "SESSIONS", "1:N", "b", "t")
    poly_arrow(d, [(80, 300), (40, 300), (40, 980), (275, 980)], "#64748b", "1:1", width=4)
    connector("USERS", "PUSH_SUBS", "1:N", "r", "l")
    connector("USERS", "USER_CCTVS", "1:N", "r", "l")
    connector("USER_CCTVS", "CCTVS", "N:1", "r", "l")
    connector("CCTVS", "RISK_EVENTS", "1:N", "b", "t")
    connector("CCTVS", "SCORE", "1:N", "r", "l")
    connector("CCTVS", "HEALTH", "1:N", "r", "l")
    save(image, "08_backend_erd.png")


def render_backend_push() -> None:
    image, d = new_canvas("백엔드 이벤트 저장 및 푸시 알림 로직")
    box(d, 80, 170, 260, 105, "추론 서버\n/inference/events", "blue", 25)
    box(d, 460, 170, 260, 105, "Token 검증\nCCTV code 조회", "white", 24)
    box(d, 840, 170, 260, 105, "Score sample\n항상 저장", "cyan", 24)
    box(d, 460, 430, 280, 120, "Threshold 판단\nwarning / danger / critical", "amber", 23)
    box(d, 840, 430, 280, 120, "Incident 병합\n15분 window\npeak update", "violet", 22)
    box(d, 1250, 430, 250, 120, "SSE publish\nPWA 즉시 갱신", "green", 23)
    box(d, 460, 720, 280, 120, "Push rule\n단계별 ON/OFF\nquiet hours", "red", 22)
    box(d, 840, 720, 280, 120, "Cooldown\n10분 / peak +10", "orange", 23)
    box(d, 1250, 720, 250, 120, "Web Push\n연결 사용자 알림", "slate", 23)
    box(d, 80, 430, 260, 120, "낮은 점수\n45 미만 또는 warning 미만\nscore만 갱신", "slate", 20)
    arrow(d, 340, 222, 460, 222, "#2563eb", "payload")
    arrow(d, 740, 222, 840, 222, "#0891b2", "valid")
    arrow(d, 970, 275, 600, 430, "#d97706", "score")
    arrow(d, 740, 490, 840, 490, "#7c3aed", "기준 충족")
    arrow(d, 1120, 490, 1250, 490, "#16a34a", "event")
    arrow(d, 600, 550, 600, 720, "#dc2626", "notify 후보")
    arrow(d, 740, 780, 840, 780, "#ea580c", "허용")
    arrow(d, 1120, 780, 1250, 780, "#334155", "전송")
    arrow(d, 460, 485, 340, 490, "#64748b", "낮은 점수")
    arrow(d, 340, 490, 1250, 490, "#64748b", "score event", width=3)
    label(d, 960, 980, "STT 원문은 저장/전송하지 않고 위험 메타데이터만 사용", 25, "#dc2626")
    save(image, "09_backend_event_push_flow.png")


def main() -> None:
    render_architecture()
    render_sequence()
    render_risk_scoring()
    render_audio_privacy()
    render_pairing()
    render_webrtc()
    render_clip_pipeline()
    render_erd()
    render_backend_push()


if __name__ == "__main__":
    main()
