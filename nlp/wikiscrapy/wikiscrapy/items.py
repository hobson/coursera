# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field

class WikiItem(Item):
    url = Field()
    title = Field()
    toc = Field()
    abstract = Field()
    text = Field()
    modified = Field()
    crawled = Field()
    count = Field()
