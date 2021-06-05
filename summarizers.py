from summary.tf_idf import run_summarization_tf_idf
from summary.word_frequency import run_summarization_wf
from summa import summarizer

def summarize_textrank(text):
    return summarizer.summarize(text)

def summarize_tfidf(text):
    return run_summarization_tf_idf(text)

def summarize_wf(text):
    return run_summarization_wf(text)

if __name__ == "__main__":
    text = """Sachin Ramesh Tendulkar (/ˌsʌtʃɪn tɛnˈduːlkər/ (About this soundlisten); born 24 April 1973) is an Indian former international cricketer who served as captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket.[5] He is the highest run scorer of all time in International cricket. Considered as the world's most prolific batsman of all time,[6] he is the only player to have scored one hundred international centuries, the first batsman to score a double century in a One Day International (ODI), the holder of the record for the most runs in both Test and ODI cricket, and the only player to complete more than 30,000 runs in international cricket.[7] In 2013, he was the only Indian cricketer included in an all-time Test World XI named to mark the 150th anniversary of Wisden Cricketers' Almanack.[8][9][10] He is affectionately known as Little Master or Master Blaster.[11][12][13][14]
    Tendulkar took up cricket at the age of eleven, made his Test debut on 15 November 1989 against Pakistan in Karachi at the age of sixteen, and went on to represent Mumbai domestically and India internationally for close to twenty-four years. In 2002, halfway through his career, Wisden Cricketers' Almanack ranked him the second-greatest Test batsman of all time, behind Don Bradman, and the second-greatest ODI batsman of all time, behind Viv Richards.[15] Later in his career, Tendulkar was a part of the Indian team that won the 2011 World Cup, his first win in six World Cup appearances for India.[16] He had previously been named "Player of the Tournament" at the 2003 edition of the tournament, held in South Africa.
    Tendulkar received the Arjuna Award in 1994 for his outstanding sporting achievement, the Rajiv Gandhi Khel Ratna award in 1997, India's highest sporting honour, and the Padma Shri and Padma Vibhushan awards in 1999 and 2008, respectively, India's fourth- and second-highest civilian awards.[17] After a few hours of his final match on 16 November 2013, the Prime Minister's Office announced the decision to award him the Bharat Ratna, India's highest civilian award.[18][19] He is the youngest recipient to date and the first ever sportsperson to receive the award.[20][21] He also won the 2010 Sir Garfield Sobers Trophy for cricketer of the year at the ICC awards.[22] In 2012, Tendulkar was nominated to the Rajya Sabha, the upper house of the Parliament of India.[23] He was also the first sportsperson and the first person without an aviation background to be awarded the honorary rank of group captain by the Indian Air Force.[24] In 2012, he was named an Honorary Member of the Order of Australia.[25][26]
    In 2010, Time magazine included Sachin in its annual Time 100 list as one of the "Most Influential People in the World".[27] In December 2012, Tendulkar announced his retirement from ODIs.[28] He retired from Twenty20 cricket in October 2013[29] and subsequently retired from all forms of cricket on 16 November 2013 after playing his 200th Test match, against the West Indies in Mumbai's Wankhede Stadium.[30] Tendulkar played 664 international cricket matches in total, scoring 34,357 runs.[7]
    In 2019, Tendulkar was inducted into the ICC Cricket Hall of Fame.[31]"""

    print("*" *100)
    print("Summary using textrank")
    print(summarize_textrank(text))

    print("*" *100)
    print("Summary using tf_idf")
    print(summarize_tfidf(text))

    print("*" *100)
    print("Summary using word frequency")
    print(summarize_wf(text))

