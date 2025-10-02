import json
import re

def create_jenga(model="./data/models/rigid_cube.obj",
                 scale=[1.5, 0.5, 0.5],
                 origin=[2.5, 0.71, 2.5],
                 start_obj_id=0,
                 num_blocks_per_layer=3,
                 num_layers=3,
                 gap=0.06):
    """
    젠가 타워를 구성하는 RigidBody 객체들의 JSON 스크립트를 생성합니다.
    (회전 대신 스케일을 변경하는 방식)

    Args:
        model (str): 블록의 3D 모델 파일 경로.
        scale (list): 블록의 기본 [길이, 높이, 너비] 크기.
        origin (list): 젠가 타워의 중심 [x, y, z] 좌표.
        start_obj_id (int): 첫 번째 블록의 objectId.
        num_blocks_per_layer (int): 한 층에 쌓이는 블록의 수.
        num_layers (int): 젠가 타워의 총 층수.
        gap (float): 블록 사이의 간격.

    Returns:
        str: 젠가 타워를 설명하는 JSON 형식의 문자열.
    """
    rigid_bodies = []
    current_obj_id = start_obj_id

    # 기본 스케일 값 정의
    block_height = scale[1]
    block_width = scale[2] # 블록을 나란히 놓을 때 기준이 되는 너비

    # X축과 Z축 방향으로 길게 놓았을 때의 스케일 정의
    scale_x_aligned = [scale[0], scale[1], scale[2]]
    scale_z_aligned = [scale[2], scale[1], scale[0]]

    # 한 층에서 블록들이 옆으로 나열될 때 차지하는 총 너비 계산
    layer_span = num_blocks_per_layer * block_width + (num_blocks_per_layer - 1) * gap
    # 중앙 정렬을 위한 시작 오프셋 계산
    start_offset = -layer_span / 2.0 + block_width / 2.0

    # 각 층을 순회
    for l in range(num_layers):
        # 현재 층의 y축 높이 계산
        y_pos = origin[1] + l * (block_height + gap)

        # 짝수 층과 홀수 층의 블록 방향을 스케일로 제어
        if l % 2 == 0:
            # 짝수 층: X축 방향으로 긴 블록을 Z축을 따라 나열
            current_scale = scale_x_aligned
            x_pos = origin[0]
            for b in range(num_blocks_per_layer):
                z_pos = origin[2] + start_offset + b * (block_width + gap)
                translation = [round(x_pos, 4), round(y_pos, 4), round(z_pos, 4)]
                
                block = {
                    "objectId": current_obj_id,
                    "geometryFile": model,
                    "translation": translation,
                    "rotationAxis": [0, 1, 0],
                    "rotationAngle": 0, # 회전은 0으로 고정
                    "scale": current_scale, # X축 정렬 스케일 적용
                    "velocity": [0.0, 0.0, 0.0],
                    "density": 1000.0,
                    "color": [255, 255, 255, 1.0],
                    "isDynamic": True
                }
                rigid_bodies.append(block)
                current_obj_id += 1
        else:
            # 홀수 층: Z축 방향으로 긴 블록을 X축을 따라 나열
            current_scale = scale_z_aligned
            z_pos = origin[2]
            for b in range(num_blocks_per_layer):
                x_pos = origin[0] + start_offset + b * (block_width + gap)
                translation = [round(x_pos, 4), round(y_pos, 4), round(z_pos, 4)]

                block = {
                    "objectId": current_obj_id,
                    "geometryFile": model,
                    "translation": translation,
                    "rotationAxis": [0, 1, 0],
                    "rotationAngle": 0, # 회전은 0으로 고정
                    "scale": current_scale, # Z축 정렬 스케일 적용
                    "velocity": [0.0, 0.0, 0.0],
                    "density": 1000.0,
                    "color": [255, 255, 255, 1.0],
                    "isDynamic": True
                }
                rigid_bodies.append(block)
                current_obj_id += 1
                
    jenga_json_obj = {"RigidBodies": rigid_bodies}
    json_string_indented = json.dumps(jenga_json_obj, indent=4)
    
    # 정규 표현식을 사용하여 리스트 내부의 줄바꿈만 제거
    pattern = re.compile(r'\[([^\[\]]+?)\]', re.MULTILINE | re.DOTALL)
    
    def compact_list_formatter(match):
        items = match.group(1).split(',')
        compacted_items = [item.strip() for item in items if item.strip()]
        return f"[{', '.join(compacted_items)}]"

    return pattern.sub(compact_list_formatter, json_string_indented)

# --- 예제 실행 ---
if __name__ == "__main__":
    jenga_script = create_jenga()
    print(jenga_script)