# 📦 ROX - Digital Twin Based Automated Logistics System

![ROX Logo](https://img.shields.io/badge/Project-ROX-red)
![Isaac Sim](https://img.shields.io/badge/Platform-NVIDIA%20Isaac%20Sim-green)
![Python](https://img.shields.io/badge/Language-Python%203.x-blue)

**ROX** 프로젝트는 객체 인식 기술을 활용하여 택배 박스를 지역별로 자동 분류하고, 협동 로봇과 자율 주행 카트를 연동하여 상차 공정을 자동화하는 **디지털 트윈 기반 물류 시스템**입니다.

---

## 🎥 Project Overview
- **주제**: 객체 인식 기반 자동 지역 분류 및 트럭 자동 상차 시스템
- **배경**: 택배 물량 증가에 따른 단순 반복 작업의 인력 부담 해소 및 공정 효율화
- **핵심 기능**: 카메라 비전 기반 색상 분류, UR10 로봇 제어, 자율 주행 카트 연동

[Image of an automated logistics sorting system using conveyor belts, sensors, and robotic arms to sort objects by color into different lanes for truck loading]

---

## 🛠 Tech Stack
- **Simulation**: NVIDIA Isaac Sim
- **Language**: Python
- **Motion Control**: Lula RMPFlow (UR10)
- **Version Control**: GitHub

---

## 🏗 System Architecture
시스템은 크게 인식, 제어, 구동의 세 계층으로 구성됩니다.

1. **Perception (인식)**: LiDAR를 통한 거리 탐지 및 카메라를 통한 큐브 색상(RGB) 분석
2. **Control (제어)**: 색상 정보에 따른 지역(RED, GREEN, BLUE) 분류 로직 및 상차 판단
3. **Actuation (구동)**: 
    - **UR10 Sorting Robot**: 메인 라인에서 색상별 컨베이어로 큐브 분류
    - **UR10 Loading Robots**: 분류된 큐브를 카트에 상차
    - **Wheeled Carts**: 큐브 2개 적재 시 자동 출발 및 이동

---

## 📊 Monitoring & UI
`BaseSampleUITemplate`을 확장한 커스텀 UI를 통해 실시간 공정 데이터를 확인할 수 있습니다.
- **Info Display**: 시뮬레이션 시간, 현재 분류 색상, 총 처리량 표시
- **Cart Status**: 각 카트(지역별)에 적재된 큐브 개수 모니터링
- **Logging**: 모든 공정 결과는 `rox_data.txt`에 실시간 기록

[Image of an Isaac Sim custom UI window with data labels and debugging information]

---

## 🚀 Getting Started
### Prerequisites
- NVIDIA Isaac Sim 설치 환경
- Python 3.x 환경

### Installation
1. 이 저장소를 클론합니다.
   ```bash
   git clone [https://github.com/your-repo/rox-project.git](https://github.com/your-repo/rox-project.git)
   ```
2. Isaac Sim의 Extension Manager에서 프로젝트 폴더를 추가하거나, 제공된 rox_project_extension.py를 실행합니다

---

## 👥 Team: E-3조 (ROX)

* **황익주**: 공정 동작 구현 
* **박제준**: 디버깅(Debug) 및 데이터 관리 
* **배민혁**: USD 환경 구축 및 모델링 
* **조재범**: 공정 동작 구현 
* **이승준**: 사용자 인터페이스(UI) 개발


K-Digital Training [두산로보틱스] 지능형 로보틱스 엔지니어 과정
