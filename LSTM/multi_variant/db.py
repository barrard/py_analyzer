from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint

import datetime
# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient('localhost',27017)
db=client.iex_stock_app
# Issue the serverStatus command and print the results
# serverStatusResult=db.command("serverStatus")
# pprint(serverStatusResult)


""" minutely_commodities is the collection with the minutly commodity data """
minutely_commodities = db.minutely_commodities
trades = db.trades



""" insert documents """
trade = {"symbol": "/ES",
         "price": 2800,
         "position":"long",
         "quantiy":2,
         "stop": 2790,
         "target":2815,
         "date": datetime.datetime.utcnow()}

# trade_id = trades.insert_one(trade).inserted_id


# pprint(trade_id)

# pprint(trades.find_one({"symbol":"/ES"}))


from bson.objectid import ObjectId

def get_document(post_id):
    # Convert from string to ObjectId:
    document = client.db.collection.find_one({'_id': ObjectId(post_id)})

def get_symbol(symbol, count=1000):
    # Convert from string to ObjectId:
    document = list(minutely_commodities
    .find({'symbol': symbol},
    {"_id":0, "close":1, "open":1, "high":1, 'low':1, "volume":1, "start_timestamp":1}
    )
    .sort([('_id',-1)])#to get the newest docuemnts
    .limit(count)
    )
    # pprint(document)
    document.reverse()#order from oldest to newest

    return document


