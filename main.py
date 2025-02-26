import cv2
from ultralytics import YOLO

# 모델 경로
model = YOLO(r"best.pt")

cap = cv2.VideoCapture(0)

width, height = 640, 480  # 화면 크기 고정
center_x, center_y = int(width / 2), int(height / 2)
target_x1, target_y1 = center_x - 50, height - 200
target_x2, target_y2 = center_x + 50, height

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    results = model(frame, conf=0.7)
    annotated_frame = results[0].plot()

    # 출력화면 정보 표시용
    cv2.rectangle(annotated_frame, (0, 0), (width, 30), (0, 0, 0), -1)  # 출력화면의 상단 텍스트 박스
    cv2.line(annotated_frame, (0, center_y), (width, center_y), (0, 0, 0), 2)  # 출력화면의 x축
    cv2.line(annotated_frame, (center_x, 0), (center_x, height), (0, 0, 0), 2)  # 출력화면의 y축

    # 여러 바운딩 박스 중 하나(최종 목표)를 택하기 위한 변수 추가
    highest_conf = 0.0  # 현재는 신뢰도가 가장 높은 것을 기준으로 하기에 max 판별용 변수
    best_pet_box = None  # 최종 목표 바운딩 박스 저장용 변수

    # 바운딩 박스 좌표
    state = ""
    for result in results:  # results 내에 여러 result 존재 가능성이 있기에 for문
        boxes = result.boxes  # 바운딩 박스(들) 정보
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌상단: (x1, y1), 우하단: (x2, y2)
            x, y = (x1 + x2) / 2, (y1 + y2) / 2  # 중앙 좌표

            # 모델에 따라 "PET", "Bottle", "PET Bottle" 등 라벨이 다를 수 있으므로 class_name if문 수정 필요
            cls_id = int(box.cls[0])  # 모델이 예측한 클래스 ID(ex. 같은 PET를 인식한 경우 ID가 동일)
            class_name = model.names[cls_id]  # model에 등록된 ID의 class 이름
            conf = float(box.conf[0])  # 해당 바운딩 박스의 신뢰도

            # 우리는 PET만 인식할 것이기에 if문으로 PET일 경우만 저장
            if class_name == "PET Bottle":  # 라벨이 달라질 경우 수정
                if conf > highest_conf:  # 최종 목표 판별
                    highest_conf = conf
                    best_pet_box = box

    # if (len(results[0].boxes) == 0):
    #    cv2.putText(annotated_frame, "No Object", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA) # 텍스트

    # 최종 목표는 빨간색 박스로 화면에 표시
    if best_pet_box is not None:  ### 수정 부분
        bx1, by1, bx2, by2 = map(int, best_pet_box.xyxy[0])
        bx, by = (bx1 + bx2) / 2, (by1 + by2) / 2
        cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)  # 빨간색 테두리
        cv2.putText(
            annotated_frame,
            f"Highest PET: {highest_conf:.2f}",
            (bx1, max(by1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # 촬영화면을 기준으로 최종목표를 타겟 박스(화면 중앙)에 위치하기 위해 필요한 로봇의 움직임
        if (bx <= target_x1):
            state = "Move Left"
        elif (bx >= target_x2):
            state = "Move Right"
        else:
            if (by <= target_y1):
                state = "Move Forward"
            else:
                state = "In Target Box"

        cv2.putText(annotated_frame, f"pos: {bx, by}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)  # 화면 텍스트
        cv2.putText(annotated_frame, f"state: {state}", (center_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    2, cv2.LINE_AA)  # 화면 텍스트
    else:  # 최종목표가 인식이 안 되었을 경우
        cv2.putText(annotated_frame, "No Object", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)  # 텍스트

    # 타겟 범위
    cv2.rectangle(annotated_frame, (target_x1, target_y1), (target_x2, target_y2), (0, 255, 0), 3)  # 타겟 박스

    cv2.imshow("화면", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()