import scrapy
import os
import re
# original text from that website is in windows-1252 coding

class SummaspiderSpider(scrapy.Spider):
    name = 'summaSpider'
    allowed_domains = []
    start_urls = ['https://oyc.yale.edu/courses/']

    def parse(self, response):
        print('--------------------------in parse')
        for link in response.css('td.views-field.views-field-field-course-number a::attr(href)'):
            # 
            yield response.follow(link.get(), callback=self.parse_courses)

    def parse_courses(self, response):
        print('--------------------------in parse_courses')
        
        for link in response.css('.views-field.views-field-field-session-display-title a::attr(href)'):
            # print(link.get())
            yield response.follow(link.get(), callback=self.download_transcript)

    def download_transcript(self, response):    
        print('--------------------------in download_transcript')
        try:
            department, course, lecture = response.url.split('https://oyc.yale.edu/')[1].split('/')
            cur_path='./'+'dataset'
            if not os.path.isdir(cur_path):
                os.mkdir(cur_path)
            cur_path+=('/'+ department)
            if not os.path.isdir(cur_path):
                os.mkdir(cur_path)
            cur_path+=('/'+ course+'/')
            if not os.path.isdir(cur_path):
                os.mkdir(cur_path)
                
            #summary(Overview)
            with open(cur_path+lecture+'_overview.txt', 'w', encoding='utf-8')as f:
                sentences = response.css('.session-title::text').getall()
                sentences = map(lambda s: re.sub('\[.+\]','',s), sentences)
                f.writelines(sentences)  
                f.write('\n')
                sentences = response.css('div.ds-1col p::text').getall()
                sentences = map(lambda s: re.sub('\[.+\]','',s), sentences)
                f.writelines(sentences)  
            #transcript
            with open(cur_path+lecture+'_transcript.txt', 'w', encoding='utf-8')as f:
                sentences = response.css('table.views-table.cols-4 p::text').getall()
                sentences = map(lambda s: re.sub('\[.+\]','',s), sentences)
                f.writelines(sentences)                

        except:
            print('--------------------------------------error:', response.url)