import os
import glob
import pandas as pd
import numpy as np

# 사용자님이 제공한 VsProfile 클래스를 정의한 모듈에서 클래스를 가져옵니다.
from vs_calc import VsProfile
from vs_calc import utils # calc_vsz에서 utils.convert_to_midpoint를 사용하므로 필요합니다.


def calculate_vsz_from_csvs(target_depth=15, source_dir="Vs", output_filename="VsZ_Results_15m_FIXED.csv"):
  """
  폴더 내의 CSV 파일들을 읽고, VsProfile을 생성하여 VsZ 값을 계산하고
  결과를 하나의 CSV 파일로 저장합니다.
  
  **[수정 사항]** VsProfile 객체 생성 시, 전체 깊이가 아닌 15m까지만 자른 데이터를 사용합니다.

  Args:
    target_depth (int): 계산할 목표 깊이 Z (VsZ에서 Z). 기본값은 15m.
    source_dir (str): CSV 파일들이 있는 폴더 경로.
    output_filename (str): 결과를 저장할 CSV 파일 이름.
  """
 
  # 결과 데이터를 저장할 리스트
  results = []

  # 1. 대상 폴더에서 모든 CSV 파일을 찾습니다.
  csv_files = glob.glob(os.path.join(source_dir, "*.csv"))
 
  if not csv_files:
    print(f"❌ '{source_dir}' 폴더에서 CSV 파일이 발견되지 않았습니다.")
    return

  print(f"🔍 총 {len(csv_files)}개의 CSV 파일을 처리합니다.")

  # 2. 각 CSV 파일 처리
  for file_path in csv_files:
    file_name = os.path.basename(file_path)
    profile_name = os.path.splitext(file_name)[0]

    try:
      # CSV 파일을 읽습니다.
      df = pd.read_csv(file_path)
     
      # VsProfile 생성자에 맞게 데이터 준비
      depth_original = np.asarray(df['d']) if 'd' in df.columns else np.asarray(df['Depth'])
      vs_original = np.asarray(df['vs']) if 'vs' in df.columns else np.asarray(df['Vs'])
     
      # Vs_SD 처리: 없으면 0으로 간주
      if 'Vs_SD' in df.columns:
        vs_sd_original = np.asarray(df['Vs_SD'])
      else:
        print(f"⚠️ 경고: {file_name} 파일에 'Vs_SD' 열이 없어 0으로 설정합니다.")
        vs_sd_original = np.zeros_like(vs_original)
     
      # --- 3. 데이터 크롭 로직 추가 (VsZ_with_smearing와 동일하게 15m로 제한) ---
      max_z = target_depth # 15m
      crop_mask = depth_original <= max_z
     
      # 15m까지 데이터가 없으면 건너뜁니다.
      if sum(crop_mask) == 0:
        print(f"⚠️ {file_name}: 15m까지의 깊이 데이터가 충분하지 않습니다. 건너뜁니다.")
        continue

      # 크롭된 데이터 정의 (VsProfile에 전달할 최종 데이터)
      depth_cropped = depth_original[crop_mask]
      vs_cropped = vs_original[crop_mask]
      vs_sd_cropped = vs_sd_original[crop_mask]
     
      # 4. VsProfile 인스턴스 생성 (크롭된 데이터 사용)
      temp_profile = VsProfile(
        name=profile_name,
        vs=vs_cropped,        # 수정: 크롭된 Vs 사용
        vs_sd=vs_sd_cropped,  # 수정: 크롭된 Vs_SD 사용
        depth=depth_cropped,  # 수정: 크롭된 Depth 사용
      )
     
      # 5. VsProfile 인스턴스 수정 및 VsZ (Z=15m) 계산
      temp_profile.max_depth = max_z
     
      # VsZ 계산
      vsz_15m = temp_profile.calc_vsz()

      # 6. 결과 저장
      results.append({
        "Profile_Name": profile_name,
        "File_Name": file_name,
        f"VsZ_{target_depth}m": round(vsz_15m, 3),
        "Max_Depth_In_File": np.max(depth_original),
        "Used_Depth_For_VsZ": temp_profile.max_depth # 15
      })
      print(f"✅ {file_name}: VsZ ({target_depth}m) = {vsz_15m:.3f} m/s 계산 완료")

    except KeyError as e:
      print(f"❌ {file_name}: 필수 열이 부족합니다. ({e}). 'd', 'vs', 'Vs_SD' 열이 필요합니다.")
    except Exception as e:
      print(f"❌ {file_name}: 처리 중 예기치 않은 오류 발생: {e}")

  # 7. 결과를 통합 CSV 파일로 저장
  if results:
    results_df = pd.DataFrame(results)
    output_path = os.path.join(source_dir, output_filename)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n🎉 모든 계산 결과가 '{output_path}'에 저장되었습니다.")
  else:
    print("\n최종 결과를 저장할 파일이 없습니다.")

if __name__ == "__main__":
  # 출력 파일명을 변경하여 기존 파일과 충돌을 피했습니다.
  calculate_vsz_from_csvs(target_depth=15, output_filename="VsZ_Results_15m_FIXED.csv")