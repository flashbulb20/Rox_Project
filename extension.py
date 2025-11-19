import omni.ext
import omni.ui as ui
import omni.timeline
import omni.kit.app
import carb
from typing import Dict

# ----------------------------------------------
# 이전 단계에서 작성한 DetailedInfoPanel 클래스를 여기에 전체 붙여넣습니다.
# (on_update, simulate_detection_logic, update_ui_elements, destroy 메소드 포함)
# ----------------------------------------------
class DetailedInfoPanel:
    # ... (이전에 만든 DetailedInfoPanel 클래스 내용 전체) ...
    def __init__(self):
        self.data: Dict[str, any] = {
            "is_playing": False,
            "current_time": 0.0,
            "last_detected_color": "None",
            "classification_id": "00",
            "total_count": 0,
            "red_count": 0,
            "blue_count": 0,
            "green_count": 0,
        }
        
        # NOTE: 창의 크기와 Docking 설정은 필요에 따라 조정하세요.
        self.window = ui.Window(title="Sim Object Detector", width=400, height=250, allow_docking=True)
        
        with self.window.frame:
            with ui.VStack(spacing=5, height=0):
                self.status_label = ui.Label(
                    "🔴 Simulation Inactive", 
                    height=20, 
                    alignment=ui.Alignment.CENTER,
                    style={"font_size": 18, "color": 0xFF888888}
                )
                ui.Separator()
                
                # Grid를 대체했던 안정적인 HStack 레이아웃 사용
                with ui.Frame(height=0): 
                    with ui.HStack(spacing=5): 
                        ui.Label("Time:", width=150, alignment=ui.Alignment.RIGHT)
                        self.time_label = ui.Label("--", width=150, alignment=ui.Alignment.LEFT)

                    with ui.HStack(spacing=5):
                        ui.Label("Color:", width=150, alignment=ui.Alignment.RIGHT)
                        self.color_label = ui.Label("--", width=150, alignment=ui.Alignment.LEFT)
                        
                    with ui.HStack(spacing=5):
                        ui.Label("Class ID:", width=150, alignment=ui.Alignment.RIGHT)
                        self.id_label = ui.Label("--", width=150, alignment=ui.Alignment.LEFT)
                
                ui.Separator()

                with ui.Frame(height=0):
                    with ui.HStack(spacing=5):
                        ui.Label("Total Processed:", width=150, alignment=ui.Alignment.RIGHT)
                        self.total_count_label = ui.Label("0", width=150, alignment=ui.Alignment.LEFT)
                        
                    with ui.HStack(spacing=5):
                        ui.Label("R/G/B Count:", width=150, alignment=ui.Alignment.RIGHT)
                        self.rgb_count_label = ui.Label("R:0 / G:0 / B:0", width=150, alignment=ui.Alignment.LEFT)


        self.timeline = omni.timeline.get_timeline_interface()
        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self.on_update, name="DetailedInfoPanel Update"
        )
        
        self.update_ui_elements()
        carb.log_info("Detailed Info Panel Initialized.")

    # 4. (예시) 더미 감지 로직 - 실제 시뮬레이션 데이터로 대체 필요
    def simulate_detection_logic(self):
        frame = int(self.data["current_time"] * 60)
        if frame % 100 == 0:
            self.data["total_count"] += 1
            current_count = self.data["total_count"]
            
            if current_count % 3 == 1:
                self.data["last_detected_color"] = "Red"
                self.data["classification_id"] = "R-101"
                self.data["red_count"] += 1
            elif current_count % 3 == 2:
                self.data["last_detected_color"] = "Green"
                self.data["classification_id"] = "G-202"
                self.data["green_count"] += 1
            else:
                self.data["last_detected_color"] = "Blue"
                self.data["classification_id"] = "B-303"
                self.data["blue_count"] += 1
        else:
             pass

    # 3. 시뮬레이션 프레임 업데이트 콜백 함수
    def on_update(self, event):
        if self.timeline.is_playing():
            self.data["current_time"] = self.timeline.get_current_time()
            self.simulate_detection_logic()
        self.update_ui_elements()

    # 2. UI 위젯 업데이트 함수
    def update_ui_elements(self):
        is_playing = self.timeline.is_playing()
        
        if is_playing:
            self.status_label.text = "🟢 Simulation Running"
            self.status_label.set_style({"color": 0xFF58FA2A})
        else:
            self.status_label.text = "🔴 Simulation Paused/Stopped"
            self.status_label.set_style({"color": 0xFFFA2A2A})
            
        self.time_label.text = f"{self.data['current_time']:.2f} s"

        color = self.data['last_detected_color']
        class_id = self.data['classification_id']
        
        color_style = {"color": 0xFFFFFFFF}
        if color == "Red":
            color_style = {"color": 0xFFFF5555}
        elif color == "Green":
            color_style = {"color": 0xFF55FF55}
        elif color == "Blue":
            color_style = {"color": 0xFF5555FF}
            
        self.color_label.text = color
        self.color_label.set_style(color_style)
        self.id_label.text = class_id

        self.total_count_label.text = str(self.data['total_count'])
        self.rgb_count_label.text = (
            f"R:{self.data['red_count']} / G:{self.data['green_count']} / B:{self.data['blue_count']}"
        )

    # 5. 창 종료 및 구독 해지
    def destroy(self):
        self.subscription = None
        self.window.destroy()
        
# ----------------------------------------------


# 확장 기능의 시작점
class MyInfoDisplayToolExtension(omni.ext.IExt):
    # 확장 기능이 활성화될 때 (Isaac Sim 시작 시)
    def on_startup(self, ext_id):
        carb.log_info(f"Extension {ext_id} startup")
        # DetailedInfoPanel 인스턴스를 생성하여 창을 띄웁니다.
        self._panel = DetailedInfoPanel()

    # 확장 기능이 비활성화될 때 (Isaac Sim 종료 시 또는 사용자가 확장 기능을 끌 때)
    def on_shutdown(self):
        carb.log_info("Extension shutdown")
        # 생성된 패널 객체를 정리하고 창을 닫습니다.
        if self._panel:
            self._panel.destroy()
            self._panel = None
