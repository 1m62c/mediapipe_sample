import cv2
import mediapipe as mp
import random
import time

# MediaPipeのセットアップ
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hand Trackingのインスタンス生成
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# グーチョキパーを判定する関数
def classify_hand(landmarks):
    # 各指先と指の根元のY座標を取得
    thumb_tip = landmarks[4].y
    thumb_mcp = landmarks[2].y  # 親指の根元

    index_tip = landmarks[8].y
    index_mcp = landmarks[5].y  # 人差し指の根元

    middle_tip = landmarks[12].y
    middle_mcp = landmarks[9].y  # 中指の根元

    ring_tip = landmarks[16].y
    ring_mcp = landmarks[13].y  # 薬指の根元

    pinky_tip = landmarks[20].y
    pinky_mcp = landmarks[17].y  # 小指の根元

    # 条件を調整して判定
    # グー: 指先がすべて根元より下（曲がっている）
    if all(tip > mcp for tip, mcp in [(index_tip, index_mcp), (middle_tip, middle_mcp), (ring_tip, ring_mcp), (pinky_tip, pinky_mcp)]):
        return "Rock"
    # チョキ: 人差し指と中指が根元より上（伸びている）で、他の指は曲がっている
    elif (index_tip < index_mcp and middle_tip < middle_mcp) and \
         (ring_tip > ring_mcp and pinky_tip > pinky_mcp):
        return "Scissors"
    # パー: 指先がすべて根元より上（伸びている）
    elif all(tip < mcp for tip, mcp in [(index_tip, index_mcp), (middle_tip, middle_mcp), (ring_tip, ring_mcp), (pinky_tip, pinky_mcp)]):
        return "Paper"
    else:
        return "Unknown"

# コンピューターの手をランダムに生成する関数
def generate_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

# 勝敗判定関数
def decide_winner(user, computer):
    if user == computer:
        return "Draw"
    elif (user == "Rock" and computer == "Scissors") or \
         (user == "Scissors" and computer == "Paper") or \
         (user == "Paper" and computer == "Rock"):
        return "You Win!"
    else:
        return "You Lose!"

# カメラ映像の取得
cap = cv2.VideoCapture(0)

# タイマー設定
start_time = time.time()
interval = 10  # 秒単位で設定

# 初期設定
computer_choice = generate_computer_choice()
game_result = "Make your move"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # フレームをRGBに変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    user_result = "Unknown"

    # 手が検出された場合
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = hand_landmarks.landmark

            user_result = classify_hand(landmarks)

    # 残り時間を計算
    current_time = time.time()
    elapsed_time = current_time - start_time
    countdown = max(0, int(interval - elapsed_time))

    # カウントダウンが0になったら結果を更新
    if countdown == 0:
        computer_choice = generate_computer_choice()
        game_result = decide_winner(user_result, computer_choice)
        start_time = current_time

    # 結果を画面に表示
    cv2.putText(frame, f"Your Move: {user_result}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Computer: {computer_choice}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Result: {game_result}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Time Left: {countdown}s", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Janken Game', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()