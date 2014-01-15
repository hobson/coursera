WikiScrapy

https://github.com/hobsonlane/coursera/tree/master/nlp

##Features

`wikiscrapy/wiki_spider.py` has the following features

* enables throttling middleware so that wikipedia politeness triggers don't block or redirect queries
* enables "obey robot" but spoofs a desktop browser, to avoid being redirected to mobile pages (en.m.wikipedia.org)
* generates a word-count vector (using coursera/nlp/strutil.py and collections.Counter) and stores it as a dict

##Dependencies

WikiScrapy and the coursera/nlp packages depend on Scrapy and various other packages listed in coursera/requirements.txt. You can install them with pip

    #!/usr/bin/env bash
    pip install -r ~/src/coursera/requirements.txt

You might get lucky an all you need is Scrapy:

    pip install Scrapy

##Starting

If you have installed the coursera folder and all the required packages, you should be able to run the following shell commands from the `coursera/nlp/wikiscrapy` folder:

    #!/usr/bin/env bash
    # cd ~/src/coursera/nlp/wikiscrapy
    scrapy crawl wiki -o wikipedia_pages.json -t json 2>crawler.log &

##Monitoring

The wikipedia crawler is now running in the background. You can monitor it by looking at the tail of the log file:

    #!/usr/bin/env bash
    # cd ~/src/coursera/nlp/wikiscrapy
    tail crawler.log

##Data Collected

You can see the data it has collected in the json file

    tail wikipedia_pages.json

Back in a shell console you can control the webservice attached to the crawler:

    #!/usr/bin/env bash
    # cd ~/src/coursera/nlp
    scrapy-ws.py help
    scrapy-ws.py get-global-stats
    scrapy-ws.py get-spider-stats wiki  # <-- the name of the spider/crawler is "wiki"

##Using the Data in Python

You can load the scraped text and metadata into RAM using python: 

    #!/usr/bin/env python
    import json
    data = json.load(open('wikipedia_pages.json', 'r'))
    # data about the root page (first page downloaded) can be displayed with
    print dict(data.items()[0])

##Stopping

If you don't stop the crawler it will download about a GB a day (1 query a second to wikipedia)
To terminate the json file with a "]",  stop the crawler using the webservice

    #!/usr/bin/env bash
    # cd ~/src/coursera/nlp/
    scrapy-ws.py stop wiki

Or you can just manually  stop the background crawler task from the shell console:

    #!/usr/bin/env bash
    sudo pkill scrapy