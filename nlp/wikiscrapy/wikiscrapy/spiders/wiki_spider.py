from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.selector import Selector
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor

from wikiscrapy.items import WikiItem
from datetime import datetime
from collections import Counter

from nlp.strutil import get_words, clean_wiki_datetime

import re

RE_WIKIPEDIA_SPECIAL = re.compile(r'.*wikipedia[.]org/wiki/[^:]+[:].*')

#from time import sleep

class WikiSpider(CrawlSpider):
    """Crawls wikipedia starting at the seed page. 

    Rate limited by obeying robots.txt (see settings.py), autothrottle, and default delay of 1.1

    $ cd nlp/wikiscrapy
    $ scrapy crawl wiki -o wikipedia_erdos.json -t json

    >>> import subprocess
    >>> subprocess.check_output('scrapy', 'crawl', 'wiki', stderr=subprocess.STDOUT)

    """

    verbosity = 1
    name = 'wiki'
    download_delay = 1.1
    allowed_domains = ['en.wikipedia.org', 'en.wiktionary.org']  # , 'en.m.wikipedia.org']
    start_urls = ['''https://en.wikipedia.org/wiki/Paul_Erd%C5%91s''']
    rules = [
        Rule(SgmlLinkExtractor(allow=['/wiki/.+']), follow=True, process_links='filter_links', callback='parse_response'),
        #Rule(SgmlLinkExtractor(allow=['/wiki/.*']), 'parse_response')]
        ]

    def __init__(self, start_urls=None, *args, **kwargs):
        self.start_urls = ['''https://en.wikipedia.org/wiki/Paul_Erd%C5%91s''']
        if start_urls:
            if isinstance(start_urls, basestring):
                self.start_urls = [start_urls]
            else:
                self.start_urls = list(start_urls)
        super(WikiSpider, self).__init__(*args, **kwargs)


    def clean_list(self, l):
        ans = ['']
        for item in l:
            # TODO: regex to see if it's a number of the form 1.2.3 before creating a new line item
            # and use the section number as a key or value in a dictionary
            stripped = item.strip()
            if stripped:
                ans[-1] += stripped + ' '
            if item.endswith('\n'):
                ans[-1] = ans[-1].strip()
                ans += ['']
        return ans


    def filter_links(self, links):
        filtered_list = []
        for link in links:
            if not RE_WIKIPEDIA_SPECIAL.match(link.url):
                filtered_list += [link]
        if self.verbosity > 1:
            print '-'*20 + ' LINKS ' + '-'*20
            print '\n'.join(link.url for link in filtered_list)
        # sleep(1.1)
        if self.verbosity > 1:
            print '-'*20 + '-------' + '-'*20
        return filtered_list

    def parse_response(self, response):
        # TODO: 
        #   1. check for error pages and slowdown or halt crawling
        #   2. throttle based on robots.txt
        #   3. save to database (so that json doesn't have to be loaded manually)
        #   4. use django Models rather than scrapy.Item model
        #   5. incorporate into a django app (or make it a django app configurable through a web interface)
        #   6. incrementally build occurrence matrix rather than saving raw data to django/postgres db
        if self.verbosity > 1:
            print '='*20 + ' PARSE ' + '='*20
        sel = Selector(response)
        a = WikiItem()
        a['url'] = response.url
        a['title'] = ' '.join(sel.xpath("//h1[@id='firstHeading']//text()").extract())
        a['toc'] = ' '.join(self.clean_list(sel.xpath("//div[@id='toc']//ul//text()").extract()))
        a['text'] = ' '.join(sel.xpath('//div[@id="mw-content-text"]//text()').extract())
        a['modified'] = clean_wiki_datetime(sel.xpath('//li[@id="footer-info-lastmod"]/text()').re(r'([0-9]+\s*\w*)'))
        a['crawled'] = datetime.now()
        a['count'] = dict(Counter(get_words(a['text'])))
        if self.verbosity > 1:
            print '='*20 + '=======' + '='*20
        yield a
