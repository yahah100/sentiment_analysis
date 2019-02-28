from bs4 import BeautifulSoup as sp
from selenium import webdriver
import re
import time
import pandas as pd
from selenium.webdriver.common.by import By
#from selenium.webdriver.support import expected_conditions as EC


main_side = 'https://www.drugs.com'
start_page_link = 'https://www.drugs.com/condition/anxiety.html'

r_href = re.compile('(?<=href=").*?(?=")')
r_review = re.compile('(?<=<span>").*?(?="</span></p>\n<table)', re.DOTALL)
r_review_fix = re.compile('(?<=<span>").*?(?=$|<)', re.DOTALL)
r_rating = re.compile('(?<=rating-score">).*?(?=<)')


# Webdriver Einstellungen
chromeOption = webdriver.ChromeOptions()


chromeOption.add_argument("--headless")
chromeOption.add_argument("--no-sandbox")
chromeOption.add_argument('window-size=1920x1080')
chromeOption.add_argument("--disable-breakpad")
chromeOption.add_argument("--disable-client-side-phishing-detection")
chromeOption.add_argument("--disable-cast")
chromeOption.add_argument("--disable-cast-streaming-hw-encoding")
chromeOption.add_argument("--disable-cloud-import")
chromeOption.add_argument("--disable-popup-blocking")
chromeOption.add_argument("--ignore-certificate-errors")
chromeOption.add_argument("--disable-session-crashed-bubble")
chromeOption.add_argument("--disable-ipv6")
chromeOption.add_argument("--allow-http-screen-capture")


driver = webdriver.Chrome(chrome_options=chromeOption)


def get_pages_of_medi(new_pages, page_link, debug=False):
    """
    Funktion zum abgreifen der Links, aus dem Hauptverzeichnis der Medikamente von A-Z

    :param new_pages: Liste der neuen Seiten
    :param page_link: String link zu Seite oder Token END
    :param debug: bool print mehr Infos
    :return: new_pages, page_link
    """
    driver.get(page_link)

    html = driver.execute_script('return document.documentElement.outerHTML')
    soup = sp(html, 'html.parser')

    table = soup.findAll('td', attrs={'class': 'condition-table__reviews valign-middle'})
    next_page_raw = soup.findAll('td', attrs={'class': 'paging-list-next'})

    hrefs = []
    for data in table:
        hrefs.append(r_href.findall(str(data)))

    next_page = r_href.findall(str(next_page_raw))

    if len(next_page):
        next_page = main_side + next_page[0]
    else:
        next_page = "END"

    for href in hrefs:
        string = str(href)
        string = string[2:]
        string = string[:-2]
        string = main_side + string
        new_pages.append(string)
        # print(string)

    if debug:
        print("--- Debug ---")
        print("next page: ", next_page)

    return new_pages, next_page


new_pages = []
med_pages, next_med_page = get_pages_of_medi(new_pages, start_page_link, debug=True)

while next_med_page is not "END":
    time.sleep(1.5)
    temp_med_pages, next_med_page = get_pages_of_medi(med_pages, next_med_page, debug=True)


def get_data_from_one_med(page, data_frame_reviews, debug=False):
    """
    Grabbt die Daten einer Seite und sucht unten auf der Website die nächste Seite Bewertungen heraus
    <1 2 |3| >, falls es keine weiter Seite zu diesem Medikament gibt wird der Token End gesetzt

    :param page: current Page oder Token END
    :param data_frame_reviews: pandas Dataframe mit allen Reviews
    :param debug: bool print mehr Infos
    :return: next_page, data_frame_reviews
    """
    driver.get(page)

    html = driver.execute_script('return document.documentElement.outerHTML')
    soup = sp(html, 'html.parser')

    usr_comments = soup.findAll('div', attrs={'class': 'user-comment'})
    next_page_raw = soup.findAll('td', attrs={'class': 'paging-list-next'})

    next_page = r_href.findall(str(next_page_raw))
    if len(next_page):
        next_page = main_side + next_page[0]
    else:
        next_page = "END"

    text_review = r_review.findall(str(usr_comments))
    text_rating = r_rating.findall(str(usr_comments))

    len_rating = len(text_rating)
    len_review = len(text_review)
    if len_rating == len_review:
        for i in range(len_rating):
            int_rating = int(float(text_rating[i]))
            if r_review_fix.search(text_review[i]):
                temp_review = r_review_fix.findall(str(text_review[i]))
                text_review[i] = temp_review[(len(temp_review) - 1)]
            data_frame_reviews = data_frame_reviews.append({"rating": int_rating, "review": text_review[i]},
                                                           ignore_index=True)

    if debug:
        print("--- Debug ---")
        print("next page: ", next_page)
        print("len_rating: ", len_rating)
        print("len_review: ", len_review)

    return next_page, data_frame_reviews


# geht urch alle Medikamente durch und speichert sie im Dataframe
data_frame_reviews = pd.DataFrame()

for page in med_pages:
    n_page, data_frame_reviews = get_data_from_one_med(page, data_frame_reviews, debug=True)
    while(n_page is not "END"):
        time.sleep(1.5)
        n_page, data_frame_reviews = get_data_from_one_med(n_page, data_frame_reviews, debug=True)

# speichern
file_name = 'medical Data'
data_frame_reviews.to_csv(file_name, sep='\t', encoding='utf-8')