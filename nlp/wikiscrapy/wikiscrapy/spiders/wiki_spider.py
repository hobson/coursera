from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.selector import Selector
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor

from wikiscrapy.items import WikiItem
import dateutil.parser

class WikiSpider(CrawlSpider):

    name = 'wiki'
    allowed_domains = ['en.wikipedia.org', 'en.wiktionary.org']
    start_urls = ['''https://en.wikipedia.org/wiki/Paul_Erd%C5%91s''']
    rules = [Rule(SgmlLinkExtractor(allow=['/wiki/.*']), 'parse')]

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
        for i, s in enumerate(dt):
            if s.endswith('at'):
                dt[i] = dt[i][:-3]
        ans = ' '.join([s.strip() for s in dt])
        return dateutil.parser.parse(ans)

    def parse(self, response):
        sel = Selector(response)
        a = WikiItem()
        a['url'] = response.url
        a['title'] = ' '.join(sel.xpath("//h1[@id='firstHeading']//text()").extract())
        a['toc'] = ' '.join(self.clean_list(sel.xpath("//div[@id='toc']//ul//text()").extract()))
        a['text'] = ' '.join(sel.xpath('//div[@id="mw-content-text"]//text()').extract())
        a['modified'] = self.clean_datetime(sel.xpath('//li[@id="footer-info-lastmod"]/text()').re(r'([0-9]+\s*\w*)'))
        return a
