import sys
import json
from time import mktime
from datetime import datetime
import feedparser as fp
import newspaper
from newspaper import Article
from model import Models
from summarizers import summarize_textrank,summarize_tfidf,summarize_wf

data = {}
data["newspapers"] = {}
model = Models()

def parse_config(fname):
    # Loads the JSON files with news sites
    with open(fname, "r") as data_file:
        cfg = json.load(data_file)

    for company, value in cfg.items():
        if "link" not in value:
            raise ValueError(f"Configuration item {company} missing obligatory 'link'.")

    return cfg

def _handle_rss(company, value, count, limit):

    fpd = fp.parse(value["rss"])
    print(f"Downloading articles from {company}")
    news_paper = {"rss": value["rss"], "link": value["link"], "articles": []}
    for entry in fpd.entries:

        if not hasattr(entry, "published"):
            continue
        if count > limit:
            break
        article = {}
        article["link"] = entry.link
        date = entry.published_parsed
        article["published"] = datetime.fromtimestamp(mktime(date)).isoformat()
        try:
            content = Article(entry.link)
            content.download()
            content.parse()
        except Exception as err:
            print(err)
            print("continuing...")
            continue
        article["title"] = content.title
        article["text"] = content.text
        news_paper["articles"].append(article)
        print(f"{count} articles downloaded from {company}, url: {entry.link}")
        count = count + 1
    return count, news_paper

def _handle_fallback(company, value, count, limit):

    print(f"Building site for {company}")
    paper = newspaper.build(value["link"], memoize_articles=False)
    news_paper = {"link": value["link"], "articles": []}
    none_type_count = 0
    for content in paper.articles:
        if count > limit:
            break
        try:
            content.download()
            content.parse()
        except Exception as err:
            print(err)
            print("continuing...")
            continue

        if content.publish_date is None:
            print(f"{count} Article has date of type None...")
            none_type_count = none_type_count + 1
            if none_type_count > 10:
                print("Too many noneType dates, aborting...")
                none_type_count = 0
                break
            count = count + 1
            continue
        article = {
            "title": content.title,
            "text": content.text,
            "link": content.url,
            "published": content.publish_date.isoformat(),
        }
        news_paper["articles"].append(article)
        print(
            f"{count} articles downloaded from {company} using newspaper, url: {content.url}"
        )
        count = count + 1
        none_type_count = 0
    return count, news_paper

def run(config, limit=4):

    for company, value in config.items():
        count = 1
        if "rss" in value:
            count, news_paper = _handle_rss(company, value, count, limit)
        else:
            count, news_paper = _handle_fallback(company, value, count, limit)
        data["newspapers"][company] = news_paper

    return data

def main():
    try:
        config = parse_config("NewsPapers.json")
    except Exception as err:
        sys.exit(err)
    return run(config, limit=3)

def get_image(image_link):
    url= image_link
    article = Article(url, language="en") # en for English
    article.download()
    article.parse()
    article.nlp()
    return article.top_image


def get_news():
    data = main()
    bbcNews = data['newspapers']['bbc']['articles']
    cnnNews = data['newspapers']['cnn']['articles']
    foxNews = data['newspapers']['foxnews']['articles']
    nytimesNews = data['newspapers']['nytimes_international']['articles']
    washingtonpostNews = data['newspapers']['washingtonpost']['articles']

    channels = [bbcNews, cnnNews, foxNews, nytimesNews, washingtonpostNews]
    cnt1 = 1;cnt2 = 2
    for channel in channels:
        for news in channel:
            cnt2 += cnt1
            link = news['link']
            news["img_link"] = get_image(link)
            news["full_text"] = news['text']
            updated_text = news['text'].splitlines()
            news['text'] = updated_text[0]
            news['index'] = cnt2
        cnt1 +=1
    get_summaries(channels)
    return channels

def get_summaries(channels):
    for channel in channels:
        for news in channel:
            news['textrank'] = summarize_textrank(news['full_text'])
            news['tf_idf'] = summarize_tfidf(news['full_text'])
            news['wf'] = summarize_wf(news['full_text'])
            result = model.predict_truthfullness(news['full_text'])
            news['truthfullness'] = result[0]
            news['truthfullnessscore'] = result[1]

def get_titles():
    titles = " "
    channels = get_news()
    for channel in channels:
        for news in channel:
            titles += news['title'] + ","
    return titles

if __name__ == "__main__":
    print(get_news())
    print(get_titles())
    channels = get_news()
    for channel in channels:
        for news in channel:
            print(f"News : {news['text']}")
            print(f" Tf_idf : {news['tf_idf']}")
            print(f" textrank : {news['textrank']}")
            print(f" wf :{news['wf']}")




