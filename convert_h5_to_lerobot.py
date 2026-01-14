import h5py
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import logging
import shutil
import yaml # yaml 라이브러리 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 설정 ---
# 원본 .h5 파일이 있는 디렉토리
RAW_H5_DIR = Path("custom_dataset_2")
# 변환된 LeRobotDataset이 저장될 디렉토리 (자동 생성됨)
# LeRobotDataset은 이 디렉토리 안에 meta/, episodes/, videos/ 등의 구조를 가집니다.
OUTPUT_LEROBOT_DATASET_DIR = Path("custom_lerobot_dataset") # 새로운 디렉토리 이름

# LeRobotDataset의 repo_id (Hugging Face Hub에 푸시하지 않을 경우 로컬 식별자 역할)
# OUTPUT_LEROBOT_DATASET_DIR과 동일하게 설정하는 것이 일반적입니다.
REPO_ID = str(OUTPUT_LEROBOT_DATASET_DIR)

# 데모 에피소드의 FPS (초당 프레임 수)
# 사용자께서 10 FPS라고 알려주셨습니다.
DATASET_FPS = 10

# task.yaml 파일 경로
TASK_YAML_PATH = RAW_H5_DIR / "task.yaml"

# YAML 파일에서 task 이름 읽기
task_name = "default_task_name" # 기본값
if TASK_YAML_PATH.exists():
    try:
        with open(TASK_YAML_PATH, 'r', encoding='utf-8') as f:
            task_config = yaml.safe_load(f)
            # 'task_name' 키의 값을 읽어옵니다.
            task_name = task_config.get('task_name', task_name)
        logging.info(f"'{TASK_YAML_PATH}'에서 task 이름 '{task_name}'을 로드했습니다.")
    except Exception as e:
        logging.error(f"'{TASK_YAML_PATH}' 파일 읽기 중 오류 발생: {e}", exc_info=True)
        logging.warning(f"기본 task 이름을 사용합니다: {task_name}")
else:
    logging.warning(f"'{TASK_YAML_PATH}' 파일을 찾을 수 없습니다. 기본 task 이름을 사용합니다: {task_name}")


# --- LeRobotDataset Feature 정의 ---
# .h5 파일의 데이터를 LeRobotDataset의 표준 형식에 맞춰 매핑합니다.
# 데이터 타입(dtype)과 형태(shape)는 h5 파일 구조 검사 결과를 기반으로 합니다.
# 'names' 필드는 LeRobotDataset의 메타데이터에 사용됩니다.
FEATURES = {
    # 이미지 (dtype을 'image'로 변경하고, shape를 채널 우선으로 변경)
    "observation.images.head_rgb": {
        "dtype": "image", # 'image'로 변경
        "shape": (3, 480, 640), # (channels, height, width)로 변경
        "names": ["channels", "height", "width"], # names도 변경
    },

    # 통합 상태 (state)
    "observation.state": {
        "dtype": "float32",
        "shape": (3 + 2 + 24,), # base_state(3) + gripper_state(2) + robot_position(24) = 29
        "names": None, # 필요하다면 모든 개별 상태의 이름을 연결하여 정의 가능
    },
    # 통합 액션 (action)
    "action": {
        "dtype": "float32",
        "shape": (2 + 24,), # gripper_target(2) + robot_target_joints(24) = 26
        "names": None, # 필요하다면 모든 개별 액션의 이름을 연결하여 정의 가능
    },
    # is_first, is_last, is_terminal 필드는 명시적으로 정의해야 함 (codebase_investigator 결과)
    "is_first": {"dtype": "bool", "shape": (1,), "names": None},
    "is_last": {"dtype": "bool", "shape": (1,), "names": None},
    "is_terminal": {"dtype": "bool", "shape": (1,), "names": None},
    # task와 timestamp는 FEATURES에 정의하지 않음. task는 frame_data에 문자열로, timestamp는 내부적으로 관리됨.
}

# --- 변환 함수 ---
def convert_h5_to_lerobot_dataset():
    logging.info(f"'{RAW_H5_DIR}'에서 .h5 파일 변환 시작...")
    
    # 출력 디렉토리가 이미 존재하면 삭제하고 다시 생성 (데이터 충돌 방지)
    if OUTPUT_LEROBOT_DATASET_DIR.exists():
        logging.info(f"기존 데이터셋 디렉토리 '{OUTPUT_LEROBOT_DATASET_DIR}' 삭제 중...")
        shutil.rmtree(OUTPUT_LEROBOT_DATASET_DIR)
        logging.info("삭제 완료.")

    # LeRobotDataset 생성
    lerobot_dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=OUTPUT_LEROBOT_DATASET_DIR, # 여기에 root 경로 명시
        robot_type="rby1",  # 사용자의 로봇 타입에 맞게 변경
        fps=DATASET_FPS,
        features=FEATURES,
        # 로컬에만 저장하고 싶다면 push_to_hub=False (기본값)
    )

    # .h5 파일들을 찾아서 순회
    h5_files = sorted(list(RAW_H5_DIR.glob("*.h5")))
    if not h5_files:
        logging.error(f"'{RAW_H5_DIR}' 디렉토리에서 .h5 파일을 찾을 수 없습니다.")
        return

    for episode_idx, h5_path in enumerate(h5_files):
        logging.info(f"에피소드 {episode_idx+1}/{len(h5_files)}: '{h5_path.name}' 변환 시작...")
        try:
            with h5py.File(h5_path, 'r') as f:
                num_frames = f['head_rgb/image'].shape[0]

                for frame_idx in range(num_frames):
                    frame_data = {}

                    # is_first, is_last, is_terminal (LeRobotDataset 표준)
                    frame_data["is_first"] = np.array([frame_idx == 0], dtype=bool)
                    frame_data["is_last"] = np.array([frame_idx == num_frames - 1], dtype=bool)
                    frame_data["is_terminal"] = np.array([frame_idx == num_frames - 1], dtype=bool) # 데모의 마지막 프레임이 종료 상태라고 가정

                    # task 필드 추가 (문자열 그대로)
                    frame_data["task"] = task_name
                    
                    # 이미지 데이터 (HWC -> CHW 변환)
                    # head_rgb: (H, W, C) -> (C, H, W)
                    frame_data["observation.images.head_rgb"] = np.transpose(f['head_rgb/image'][frame_idx], (2, 0, 1))
                    # head_depth: (H, W) -> (1, H, W) (단일 채널 추가)


                    # 상태 데이터 (float64를 float32로 변환 후 하나로 연결)
                    base_state = f['samples/base_state'][frame_idx].astype(np.float32)
                    gripper_state = f['samples/gripper_state'][frame_idx].astype(np.float32)
                    robot_position = f['samples/robot_position'][frame_idx].astype(np.float32)
                    frame_data["observation.state"] = np.concatenate([base_state, gripper_state, robot_position])

                    # 액션 데이터 (float64를 float32로 변환 후 하나로 연결)
                    gripper_target = f['samples/gripper_target'][frame_idx].astype(np.float32)
                    robot_target_joints = f['samples/robot_target_joints'][frame_idx].astype(np.float32)
                    frame_data["action"] = np.concatenate([gripper_target, robot_target_joints])
                    
                    # 모든 데이터가 np.ndarray 형태인지 확인 (task 필드 제외)
                    for key, value in frame_data.items():
                        if key == "task": # 'task' 필드는 문자열 그대로 유지되어야 함
                            continue
                        if not isinstance(value, np.ndarray):
                            frame_data[key] = np.asarray(value)

                    lerobot_dataset.add_frame(frame_data)

                lerobot_dataset.save_episode()
                logging.info(f"'{h5_path.name}' 에피소드 저장 완료.")

        except Exception as e:
            logging.error(f"'{h5_path.name}' 처리 중 오류 발생: {e}", exc_info=True) # exc_info=True로 traceback 출력
            # 오류 발생 시 해당 에피소드 건너뛰기
            continue

    # 데이터셋 최종 마무리
    lerobot_dataset.finalize()
    logging.info(f"LeRobotDataset 변환 완료. '{OUTPUT_LEROBOT_DATASET_DIR}'에 저장되었습니다.")

if __name__ == "__main__":
    convert_h5_to_lerobot_dataset()