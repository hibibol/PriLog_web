import numpy as np
import os
import re
from pytube import YouTube
import time as tm
import cv2
from . import characters as cd
from . import after_caluculation as ac
# キャラクター名テンプレート
characters_data = np.load("PriLog_web/model/UB_name.npy")

# 時間テンプレート
sec_data = np.load("PriLog_web/model/timer_sec.npy")

# MENUテンプレート
menu_data = np.load("PriLog_web/model/menu.npy")

# ダメージレポートテンプレート
damage_menu_data = np.load("PriLog_web/model/damage_menu.npy")

# ダメージ数値テンプレート
damage_data = np.load("PriLog_web/model/damage_data.npy")

# キャラクター名一覧
characters = cd.characters_name

# 数値一覧
numbers = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

FRAME_COLS = 1280
FRAME_ROWS = 720

UB_ROI = (490, 98, 810, 132)
MIN_ROI = (1073, 22, 1089, 44)
TEN_SEC_ROI = (1091, 22, 1107, 44)
ONE_SEC_ROI = (1105, 22, 1121, 44)
MENU_ROI = (1100, 0, 1280, 90)
DAMAGE_MENU_ROI = (1040, 36, 1229, 66)
DAMAGE_DATA_ROI = (60, 54, 230, 93)

MENU_LOC = (63, 23)

DAMAGE_NUMBER_ROI = [
    (0, 0, 26, 39),
    (22, 0, 50, 39),
    (46, 0, 74, 39),
    (70, 0, 98, 39),
    (94, 0, 122, 39),
    (118, 0, 146, 39),
    (142, 0, 170, 39)
]

TIMER_MIN = 2
TIMER_TEN_SEC = 1
TIMER_SEC = 0

UB_THRESH = 0.6
TIMER_THRESH = 0.75
MENU_THRESH = 0.6
DAMAGE_THRESH = 0.7

FOUND = 1
NOT_FOUND = 0

NO_ERROR = 0
ERROR_BAD_URL = 1
ERROR_TOO_LONG = 2
ERROR_NOT_SUPPORTED = 3
ERROR_CANT_GET_MOVIE = 4
ERROR_REQUIRED_PARAM = 5

stream_dir = "PriLog_web/tmp/"
if not os.path.exists(stream_dir):
    os.mkdir(stream_dir)


def search(youtube_id):
    # ID部分の取り出し
    work_id = re.findall('.*watch(.{14})', youtube_id)
    if not work_id:
        work_id = re.findall('.youtu.be/(.{11})', youtube_id)
        if not work_id:
            return None, None, None, None, ERROR_BAD_URL
        work_id[0] = '?v=' + work_id[0]
    # Youtubeから動画を保存し保存先パスを返す
    youtube_url = 'https://www.youtube.com/watch' + work_id[0]
    try:
        yt = YouTube(youtube_url)
    except:
        return None, None, None, None, ERROR_CANT_GET_MOVIE

    movie_thumbnail = yt.thumbnail_url
    movie_length = yt.length
    if int(movie_length) > 480:
        return None, None, None, None, ERROR_TOO_LONG

    stream = yt.streams.get_by_itag("22")
    if stream is None:
        return None, None, None, None, ERROR_NOT_SUPPORTED

    movie_title = stream.title
    movie_name = tm.time()
    movie_path = stream.download(stream_dir, str(movie_name))

    return movie_path, movie_title, movie_length, movie_thumbnail, NO_ERROR


def analyze_movie(movie_path):
    # 動画解析し結果をリストで返す
    start_time = tm.time()
    video = cv2.VideoCapture(movie_path)

    frame_count = int(video.get(7))  # フレーム数を取得
    frame_rate = int(video.get(5))  # フレームレート(1フレームの時間単位はミリ秒)の取得

    frame_width = int(video.get(3))  # フレームの幅
    frame_height = int(video.get(4))  # フレームの高さ

    if frame_width != int(FRAME_COLS) or frame_height != int(FRAME_ROWS):
        video.release()
        os.remove(movie_path)

        return None, None

    n = 0.34  # n秒ごと*
    ub_interval = 0

    time_min = "1"
    time_sec10 = "3"
    time_sec1 = "0"

    menu_check = False

    min_roi = MIN_ROI
    tensec_roi = TEN_SEC_ROI
    onesec_roi = ONE_SEC_ROI
    ub_roi = UB_ROI
    damage_menu_roi = DAMAGE_MENU_ROI
    damage_data_roi = DAMAGE_DATA_ROI

    ub_data = []
    ub_data_value = []
    time_data = []
    characters_find = []

    tmp_damage = ["0", "0", "0", "0", "0", "0", "0"]
    total_damage = False

    cap_interval = int(frame_rate * n)
    skip_frame = 5 * cap_interval

    if (frame_count / frame_rate) < 600:  # 10分未満の動画しか見ない
        for i in range(frame_count):  # 動画の秒数を取得し、回す
            ret = video.grab()
            if ret is False:
                break

            if i % cap_interval is 0:
                if ((i - ub_interval) > skip_frame) or (ub_interval == 0):
                    ret, work_frame = video.read()

                    if ret is False:
                        break
                    work_frame = edit_frame(work_frame)

                    if menu_check is False:
                        menu_check, menu_loc = analyze_menu_frame(work_frame, menu_data, MENU_ROI)
                        if menu_check is True:
                            loc_diff = np.array(MENU_LOC) - np.array(menu_loc)
                            roi_diff = (loc_diff[0], loc_diff[1], loc_diff[0], loc_diff[1])
                            min_roi = np.array(MIN_ROI) - np.array(roi_diff)
                            tensec_roi = np.array(TEN_SEC_ROI) - np.array(roi_diff)
                            onesec_roi = np.array(ONE_SEC_ROI) - np.array(roi_diff)
                            ub_roi = np.array(UB_ROI) - np.array(roi_diff)
                            damage_menu_roi = np.array(DAMAGE_MENU_ROI) - np.array(roi_diff)
                            damage_data_roi = np.array(DAMAGE_DATA_ROI) - np.array(roi_diff)

                    else:
                        if time_min is "1":
                            time_min = analyze_timer_frame(work_frame, min_roi, 2, time_min)

                        time_sec10 = analyze_timer_frame(work_frame, tensec_roi, 6, time_sec10)
                        time_sec1 = analyze_timer_frame(work_frame, onesec_roi, 10, time_sec1)

                        ub_result = analyze_ub_frame(work_frame, ub_roi, time_min, time_sec10, time_sec1,
                                                     ub_data, ub_data_value, characters_find)

                        if ub_result is FOUND:
                            ub_interval = i

                        ret = analyze_menu_frame(work_frame, damage_menu_data, damage_menu_roi)[0]

                        if ret is True:
                            ret, end_frame = video.read()

                            if ret is False:
                                break

                            ret = analyze_damage_frame(end_frame, damage_data_roi, tmp_damage)
                            if ret is True:
                                total_damage = "総ダメージ " + ''.join(tmp_damage)
                            else:
                                ret = analyze_damage_frame(end_frame, DAMAGE_DATA_ROI, tmp_damage)
                                if ret is True:
                                    total_damage = "総ダメージ " + ''.join(tmp_damage)

                            break

    video.release()
    os.remove(movie_path)

    # TLに対する後処理
    debuff_value = ac.make_ub_value_list(ub_data_value, characters_find)

    time_result = tm.time() - start_time
    time_data.append("動画時間 : {:.3f}".format(frame_count / frame_rate) + "  sec")
    time_data.append("処理時間 : {:.3f}".format(time_result) + "  sec")

    return ub_data, time_data, total_damage, debuff_value


def edit_frame(frame):
    work_frame = frame

    work_frame = cv2.cvtColor(work_frame, cv2.COLOR_RGB2GRAY)
    work_frame = cv2.threshold(work_frame, 200, 255, cv2.THRESH_BINARY)[1]
    work_frame = cv2.bitwise_not(work_frame)

    return work_frame


def analyze_ub_frame(frame, roi, time_min, time_10sec, time_sec, ub_data, ub_data_value, characters_find):
    analyze_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]

    characters_num = len(characters)

    if len(characters_find) < 5:
        for j in range(characters_num):
            result_temp = cv2.matchTemplate(analyze_frame, characters_data[j], cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_temp)
            if max_val > UB_THRESH:
                ub_data.append(time_min + ":" + time_10sec + time_sec + " " + characters[j])
                ub_data_value.extend([[int(int(time_min) * 60 + int(time_10sec) * 10 + int(time_sec)), int(j)]])
                if j not in characters_find:
                    characters_find.append(j)

                return FOUND
    else:
        for j in range(5):
            result_temp = cv2.matchTemplate(analyze_frame, characters_data[characters_find[j]], cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_temp)
            if max_val > UB_THRESH:
                ub_data.append(time_min + ":" + time_10sec + time_sec + " " + characters[characters_find[j]])
                ub_data_value.extend([[(int(time_min) * 60 + int(time_10sec) * 10 + int(time_sec)),
                                       characters_find[j]]])

                return FOUND

    return NOT_FOUND


def analyze_timer_frame(frame, roi, data_num, time_data):
    analyze_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]

    for j in range(data_num):
        result_temp = cv2.matchTemplate(analyze_frame, sec_data[j], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_temp)
        if max_val > TIMER_THRESH:
            return numbers[j]

    return time_data


def analyze_menu_frame(frame, menu, roi):
    analyze_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]

    result_temp = cv2.matchTemplate(analyze_frame, menu, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_temp)
    if max_val > MENU_THRESH:
        return True, max_loc

    return False, None


def analyze_damage_frame(frame, roi, damage):
    analyze_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]

    analyze_frame = cv2.cvtColor(analyze_frame, cv2.COLOR_BGR2HSV)
    analyze_frame = cv2.inRange(analyze_frame, np.array([10, 120, 160]), np.array([40, 255, 255]))

    ret = False
    damage_num = len(damage)
    number_num = len(numbers)

    for i in range(damage_num):
        check_roi = DAMAGE_NUMBER_ROI[i]
        check_frame = analyze_frame[check_roi[1]:check_roi[3], check_roi[0]:check_roi[2]]
        tmp_damage = [0, NOT_FOUND]
        damage[i] = "?"
        for j in range(number_num):
            result_temp = cv2.matchTemplate(check_frame, damage_data[j], cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_temp)
            if max_val > DAMAGE_THRESH:
                if max_val > tmp_damage[1]:
                    tmp_damage[0] = j
                    tmp_damage[1] = max_val
                    ret = True

        if tmp_damage[1] != NOT_FOUND:
            damage[i] = str(tmp_damage[0])

    return ret

def main(Url):
    status = NO_ERROR
    msg = "OK"
    result = {}

    ret = {}
    ret["result"] = result

    # youtube動画検索/検証
    path, title, length, thumbnail, url_result = search(Url)
    status = url_result
    if url_result is ERROR_BAD_URL:
        msg = "URLはhttps://www.youtube.com/watch?v=...の形式でお願いします"
    elif url_result is ERROR_TOO_LONG:
        msg = "動画時間が長すぎるため、解析に対応しておりません"
    elif url_result is ERROR_NOT_SUPPORTED:
        msg = "非対応の動画です。「720p 1280x720」の一部の動画に対応しております"
    elif url_result is ERROR_CANT_GET_MOVIE:
        msg = "動画の取得に失敗しました。もう一度入力をお願いします"
    else :
        # TL解析
        time_line, time_data, total_damage, debuff_value = analyze_movie(path)
        result["total_damage"] = total_damage
        result["timeline"] = time_line
        result["timeline_txt"] = "\r\n".join(time_line)
        result["process_time"] = time_data
        result["debuff_value"] = debuff_value

    ret["msg"] = msg
    ret["status"] = status
    return ret

if __name__ == "__main__":
    demo_url = "https://www.youtube.com/watch?v=TGeZljb8RVE&feature=youtu.be"
    print(main(demo_url))