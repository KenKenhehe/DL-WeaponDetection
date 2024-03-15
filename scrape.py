import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import requests
import urllib.request
import os
import io
from PIL import Image

import time
wd = webdriver.Chrome()
search_keyword = "people on street"

has_weapon_folder = "HasWeapon"
no_weapon_folder = "NoWeapon"

output_folder = f"Data/{no_weapon_folder}/"
if(not os.path.exists(output_folder)):
    os.mkdir(output_folder)
search_url = f"https://www.google.com/search?q={search_keyword}&sca_esv=593217386&rlz=1C1ONGR_en-GBAU991AU991&tbm=isch&sxsrf=AM9HkKlohH319t3cpas3Ap-RLTbk8KecTw:1703314619525&source=lnms&sa=X&ved=2ahUKEwijldm2_aSDAxVhamwGHfmqAwMQ_AUoAXoECAQQAw&biw=1920&bih=963&dpr=1"

def get_img_from_google(wd:webdriver.Chrome, delay, max_img):
    def scroll_to_end(delay):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

    url = search_url
    wd.get(url)

    image_urls = set()

    current_thum_len = 0
    current_thumnail_list = []
    is_at_end = False
    previous_thumbnail_len = 0
    last_height = 0
    scroll_attemp = 0
    while current_thum_len < max_img and is_at_end == False:
        print(scroll_attemp)
        scroll_to_end(delay)
        new_height = wd.execute_script("return document.body.scrollHeight")
        thumbnails = wd.find_elements(By.CLASS_NAME, "Q4LuWd")
        current_thum_len = len(thumbnails)

        print(current_thum_len)
        print(previous_thumbnail_len)

        if(current_thum_len == previous_thumbnail_len):
            break
        if(scroll_attemp > 20):
            break
        current_thumnail_list = thumbnails
        show_more_result = wd.find_elements(By.CLASS_NAME, "LZ4I")
        
        try:
            if len(show_more_result) > 0:
                wd.implicitly_wait(2)
                ActionChains(wd).move_to_element(show_more_result[0]).click(show_more_result[0]).perform()
                time.sleep(3)
                scroll_attemp = 0

        except Exception as e:
            print(f"Exception occured: {e}")
            scroll_attemp += 1
            continue
           
        print(f"last height: {last_height}")
        print(f"newt height: {new_height}")
        if last_height == new_height:
            is_at_end = True
        else:
            last_height = new_height

        previous_thumbnail_len = current_thum_len

    #print(len(current_thumnail_list))
    error_count = 0
    img_found_count = 0
    for i in range(max_img):
        try:
            current_thumnail_list[i].click()
            time.sleep(delay * 0.5)
        except Exception as e:
            error_count += 1
            continue

        imgs = wd.find_elements(By.CLASS_NAME, "iPVvYb")
        for img in imgs:
            if img.get_attribute("src") and "http" in img.get_attribute("src"):
                image_urls.add(img.get_attribute("src"))
                print("Found image!")
                img_found_count += 1

    #print(f"Error count: {error_count} out of {max_img}")
    print(f"Found: {img_found_count} out of {max_img}")
    return image_urls    

def download_image(download_path, url, file_name):
    try:
        image_content = requests.get(url, timeout=5).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)

        file_path = download_path + file_name
        with open(file_path, "wb") as f:
            image.save(f, "JPEG")
    except Exception as e:
        print(f"{file_name} has encounter an exception: {e}")

url_list = get_img_from_google(wd, 0.3, 1000)

wd.quit()

for idx, url in enumerate(url_list):
    print(url)
    download_image(output_folder, url, str(idx) + ".jpg")