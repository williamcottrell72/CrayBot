# MVP Project

## Domain

My goal is to analyze real-life chatbot data generated on the prime.ai platform.  One goal is to apply NLP / unsupervised learning in order to group users and bots into categories.  I would then like to make several chatbots based on the conversations people have with the chatbots.  I am also interested in developing a good metric for 'craziness' and correlating this with chatbot usages.

## Utility

One particular use case is to provide good metrics to help the designers of chatbots assess their clients and correlate their comments with chat-bot use.

## Data

My data was provided by prime.ai.  It consists of about 2.5 million text messages sent by users to chatbots, as well as about 8 million other interactions wtih the chatbot.  A typical  entry looks like:

array([0, {'$oid': '5b592d20d86a92001458faa2'},
       {'$date': '2018-07-26T02:08:32.940Z'},
       {'message': {'text': 'Sexy', 'seq': 274501, 'mid': 'J4pdxhq4W3KpU_cAIY38UGBucddvBT-YCw7b9Hl8neesQWDYGUztEkOABhlHHbwDJ8mbssoJS9V4Dg_mnbP8XQ'}, 'timestamp': 1532570912527.0, 'recipient': {'id': '1748596732049183'}, 'sender': {'id': '2626588190700296'}},
       1748596732049183.0], dtype=object)

Someone went to the trouble of telling the chatbot that it was sexy.  There are also more elaborate messages such as:

"Rapture will take place any time from now. Everything hindering the rapture has been removed. Gospel has been preached almost everywhere. All the prophecies have been fulfilled. Angels physically seen and captured on camera @ Dansoman also gives a 'sign'. The devil is working very hard to occupy Christians with the things of this world so that the day will catch them unawares. Please be prepared....... "

This is pretty typical.

## Known Unknowns

The biggest known unknown is where the user exits the bot.  We don't know what the user ends up doing after chatting with the bot. I also don't really know much about the pages hosting the individual bots, though I do have the urls and could possibly do some scraping.  
