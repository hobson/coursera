from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.selector import Selector
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor

from wikiscrapy.items import WikiItem
import dateutil.parser
from time import sleep

class WikiSpider(CrawlSpider):
    """Crawls wikipedia starting at the seed page. 

    Rate limited to 1.1 second per article (per wikipedia robots.txt)
    >>> cd wikiscrapy/wikiscrapy
    >>> scrapy crawl wiki -o wikipedia_erdos.json -t json
    """

    name = 'wiki'
    allowed_domains = ['en.wikipedia.org', 'en.wiktionary.org']
    start_urls = ['''https://en.wikipedia.org/wiki/Paul_Erd%C5%91s''']
    rules = [
        Rule(SgmlLinkExtractor(allow=['/wiki/.+']), follow=True, process_links='filter_links', callback='parse_response'),
        #Rule(SgmlLinkExtractor(allow=['/wiki/.*']), 'parse_response')]
        ]

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

    def clean_datetime(self, dt):
        at_i = None
        for i, s in enumerate(dt):
            if s.endswith('at'):
                dt[i] = dt[i][:-3]
                at_i = i
        if at_i is not None and len(dt) > (at_i + 1) and dt[at_i + 1] and not ':' in dt[at_i + 1]:
            dt = dt[:at_i] + [dt[at_i + 1].strip() + ':' + dt[at_i + 2].strip()]
        ans = ' '.join([s.strip() for s in dt])
        try:
            return dateutil.parser.parse(ans)
        except Exception as e:
            from traceback import format_exc
            print format_exc(e) +  '\n^^^ Exception caught ^^^\nWARN: Failed to parse datetime string %r\n      from list of strings %r' % (ans, dt)
            return ans

    def filter_links(self, links):
        print '-'*20 + ' LINKS ' + '-'*20
        print '\n'.join(link.url for link in links)
        sleep(1.1)
        print '-'*20 + '-------' + '-'*20
        return links

    def parse_response(self, response):
        # TODO: 
        #   1. check for error pages and slowdown or halt crawling
        #   2. throttle based on robots.txt
        #   3. save to database (so that json doesn't have to be loaded manually)
        #   4. use django Models rather than scrapy.Item model
        #   5. incorporate into a django app (or make it a django app configurable through a web interface)
        #   6. incrementally build occurrence matrix rather than saving raw data to django/postgres db
        print '='*20 + ' PARSE ' + '='*20
        sel = Selector(response)
        a = WikiItem()
        a['url'] = response.url
        a['title'] = ' '.join(sel.xpath("//h1[@id='firstHeading']//text()").extract())
        a['toc'] = ' '.join(self.clean_list(sel.xpath("//div[@id='toc']//ul//text()").extract()))
        a['text'] = ' '.join(sel.xpath('//div[@id="mw-content-text"]//text()').extract())
        a['modified'] = self.clean_datetime(sel.xpath('//li[@id="footer-info-lastmod"]/text()').re(r'([0-9]+\s*\w*)'))
        sleep(1.1)
        print '='*20 + '=======' + '='*20
        yield a
