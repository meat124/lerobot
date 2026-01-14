import h5py
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# .h5 파일이 있는 디렉토리
H5_DIR = Path("custom_dataset_2")

# 검사할 파일 선택 (첫 번째 파일인 demo_0.h5)
h5_files = list(H5_DIR.glob("*.h5"))
if not h5_files:
    logging.error(f"'{H5_DIR}' 디렉토리에서 .h5 파일을 찾을 수 없습니다.")
else:
    file_to_inspect = h5_files[0]
    logging.info(f"--- {file_to_inspect} 파일 구조 검사 ---")

    def print_hdf5_structure(group, prefix=''):
        for key in group.keys():
            item = group[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset):
                try:
                    logging.info(f"키: {path}")
                    logging.info(f"  - 형태(Shape): {item.shape}")
                    logging.info(f"  - 데이터 타입(Dtype): {item.dtype}")
                    # 만약 데이터가 충분히 작다면 첫 번째 값 출력 시도
                    if item.ndim > 0 and item.shape[0] > 0 and item.nbytes < 1024:
                        logging.info(f"  - 첫 번째 값: {item[0]}")
                    elif item.ndim == 0: # 스칼라 값
                        logging.info(f"  - 값: {item[()]}")
                except Exception as e:
                    logging.warning(f"  - 키 '{path}'의 정보 출력 중 오류 발생: {e}")
            elif isinstance(item, h5py.Group):
                logging.info(f"그룹: {path}")
                print_hdf5_structure(item, prefix=path)

    try:
        with h5py.File(file_to_inspect, 'r') as f:
            print_hdf5_structure(f)
        logging.info("--- 검사 완료 ---")
    except Exception as e:
        logging.error(f"'{file_to_inspect}' 파일 열기/읽기 중 오류 발생: {e}")