<!-- 30cb1db2-d83e-469c-94f8-252109b87080 fad31921-91cc-4130-a4e6-f3330e1aa9f8 -->
# main.py 슬림화 + visualization.py / output_system.py 통합 계획

## 목표

- 단일 엔트리포인트 `main.py` 유지, 시뮬레이션은 항상 실행(핵심 루프).
- GUI로 시각화 옵션(원본 색/히트맵, 히트맵 대상, 강체 히트맵 on/off, 투명/반투명) 제어 유지.
- 출력(export)은 별도 모듈 `output_system.py`에서 관리. 강체/탄성 메쉬, 강체/탄성 파티클, 유체 파티클 export 지원. 파티클 export 시 히트맵 attribute 강제 포함. 포맷은 PLY 우선, VTK/Partio는 스텁.

## 아키텍처 개요(단순화)

- main.py: 설정/CLI 파싱 → 시뮬레이션 초기화 → 콜백 연결(Viewer, OutputManager) → 루프 실행.
- visualization.py: Viewer(UI+렌더), VisualizationSettings, `fill_visualization_buffers`(기존 `copy_to_vis_buffer` 이관).
- output_system.py: OutputManager(수집/속성/네이밍/쓰기), PLY writer(실구현), VTK/Partio writer(스텁), OutputConfig/ExportFormat.

## 파일 구조(간소화)

- `main.py` (엔트리포인트, CLI 포함)
- `visualization.py` (모든 viz 기능 통합)
- `output_system.py` (모든 export 기능 통합 + OutputManager)

## 공개 API(핵심 시그니처)

```python
# visualization.py
from dataclasses import dataclass
from enum import Enum

class ColorMode(str, Enum):
    original = "original"
    heatmap = "heatmap"

class HeatmapField(str, Enum):
    velocity = "velocity"
    density = "density"
    div_velocity = "div_velocity"

class Transparency(str, Enum):
    opaque = "opaque"
    translucent = "translucent"
    transparent = "transparent"

@dataclass
class VisualizationSettings:
    color_mode: ColorMode = ColorMode.original
    heatmap_field: HeatmapField = HeatmapField.velocity
    enable_rigid_heatmap: bool = False
    transparency: Transparency = Transparency.opaque

class Viewer:
    def __init__(self, settings: VisualizationSettings):
        self.settings = settings
        self.output_manager = None  # 선택: GUI에서 export 옵션 제어를 위해 연결

    def attach_output_manager(self, output_manager):
        self.output_manager = output_manager

    def on_step(self, frame_idx: int, sim_state) -> None:
        vis = fill_visualization_buffers(sim_state, self.settings)
        # 버퍼 업로드/렌더 호출

    def render_ui(self) -> None:
        # viz 옵션 패널 + (옵션) export 패널(OutputManager.config 바인딩)
        ...

class VisBuffers: ...

def fill_visualization_buffers(sim_state, settings: VisualizationSettings) -> VisBuffers:
    """기존 particle_system.copy_to_vis_buffer 로직 이관."""
    ...
```



```python
# output_system.py
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

class ExportFormat(str, Enum):
    ply = "ply"
    vtk = "vtk"      # 스텁
    partio = "partio"  # 스텁

@dataclass
class OutputConfig:
    root_dir: str
    export_formats: set[ExportFormat]
    export_rigid_mesh: bool = True
    export_elastic_mesh: bool = True
    export_rigid_particles: bool = True
    export_elastic_particles: bool = True
    export_fluid_particles: bool = True
    frame_interval: int = 1
    include_heatmap_attributes: bool = True
    filename_pattern: str = "{entity}_{kind}_{frame:06d}.{ext}"

class OutputManager:
    def __init__(self, config: OutputConfig):
        self._config = config

    # GUI에서 제어하기 위한 바인딩
    def get_config(self) -> OutputConfig: return self._config
    def set_config(self, cfg: OutputConfig) -> None: self._config = cfg

    def on_step(self, frame_idx: int, sim_state) -> None:
        if frame_idx % self._config.frame_interval != 0:
            return
        # 수집
        rigid_meshes, elastic_meshes = collect_meshes(sim_state)
        rigid_particles, elastic_particles, fluid_particles = collect_particles(sim_state)
        # 속성(히트맵 등) 계산 - 시각화와 무관하게 항상 포함
        attrs = build_particle_attributes(sim_state, include_heatmap=self._config.include_heatmap_attributes)
        # 쓰기
        for fmt in self._config.export_formats:
            if fmt == ExportFormat.ply:
                write_ply(frame_idx, rigid_meshes, elastic_meshes,
                          rigid_particles, elastic_particles, fluid_particles,
                          attrs, self._config)
            elif fmt == ExportFormat.vtk:
                write_vtk_stub(...)
            elif fmt == ExportFormat.partio:
                write_partio_stub(...)

    def finalize(self) -> None:
        # 열린 리소스 정리(필요 시)
        pass

# ---- helpers (동일 파일 내) ----

def collect_meshes(sim_state):
    ...

def collect_particles(sim_state):
    ...

def build_particle_attributes(sim_state, include_heatmap: bool):
    ...  # velocity/density/div_velocity 등 공통 attribute 구성

# 파일명 구성은 단순 함수로 제공
def build_filename(entity: str, frame_idx: int, kind: str, ext: str, cfg: OutputConfig) -> str:
    return cfg.filename_pattern.format(entity=entity, frame=frame_idx, kind=kind, ext=ext)

# PLY는 실제 구현, 나머지는 스텁
def write_ply(frame_idx: int, rigid_meshes, elastic_meshes,
              rigid_particles, elastic_particles, fluid_particles,
              attrs, cfg: OutputConfig) -> None:
    ...

def write_vtk_stub(*args, **kwargs):
    pass

def write_partio_stub(*args, **kwargs):
    pass
```

## main.py 변경 요약

- CLI 파싱(포맷/엔티티/출력 루트/간격/히트맵 필드 등) → `VisualizationSettings`와 `OutputConfig` 생성.
- 시뮬레이션 초기화 후, 메인 루프에서 매 스텝마다:
  - `viewer.on_step(i, state)` 호출로 화면 갱신.
  - `output_manager.on_step(i, state)` 호출로 export.
- 종료 시 `output_manager.finalize()` 호출.
- GUI 이벤트로 `viewer.render_ui()` 내에서 viz 옵션과 export 옵션을 함께 제어.

## 마이그레이션 순서(간소/안전)

1. `particle_system.copy_to_vis_buffer` → `visualization.fill_visualization_buffers`로 이관(기능 동일).
2. `Viewer` 생성: 기존 GUI 초기화/패널/렌더 호출을 `visualization.py`로 이동.
3. `OutputManager` 구현: 수집/속성/네이밍/PLY writer를 `output_system.py`에 통합.
4. main 루프에서 두 콜백(`viewer.on_step`, `output_manager.on_step`) 연결하여 동등 동작 확인.
5. CLI 옵션을 `VisualizationSettings`/`OutputConfig`로 매핑.
6. VTK/Partio는 스텁로 남기고, 추후 구현 시 교체.
7. main에서 시각화/출력 관련 로직 제거(엔트리포인트로 슬림화).
8. deprecated 애니메이션 코드 분리.

## 파일명/포맷 규칙

- `build_filename(entity, frame, kind, ext, cfg)`를 `output_system.py` 내 단일 함수로 관리.
- 포맷 선택: 기본 PLY, VTK/Partio는 GUI에서 선택해도 내부적으로 스텁 경고만 출력.

## 리스크/완화

- Viewer와 OutputManager 간 결합: `attach_output_manager`로 선택적 연결, 단방향 참조만 유지.
- 성능: 버퍼/속성 계산 경로는 기존 로직을 재사용하여 비용 유지.
- 의존성: PLY만 즉시 구현, 나머지는 의존성 추가 없이 진행 가능.

## 후속(차후)

- VTK/Partio 실제 구현과 옵션 고급화(압축, 이진/아스키 등).
- 투명/반투명 렌더 품질 개선(블렌딩/소트).
- rigid/elastic mesh export 속성 확장(법선/재질 등).

### To-dos

- [ ] SimulationRunner 추출 및 main.py 교체(동등 동작 확인)
- [ ] copy_to_vis_buffer를 viz/buffers.py로 이동하고 참조 수정
- [ ] VisualizationSettings/패널 정의 및 GUI 바인딩
- [ ] Viewer 루프/렌더 호출 분리 및 on_step 연결
- [ ] OutputManager/collectors/naming/attributes 골격 구현
- [ ] PLY writer 구현 및 프레임 export 연결
- [ ] 파티클 export 시 heatmap attribute 포함 구현
- [ ] GUI에서 포맷/엔티티/간격/출력 루트 등 제어
- [ ] main.py를 엔트리포인트/DI 조립만 남기고 정리
- [ ] deprecated 애니메이션 코드 분리/경로 정리