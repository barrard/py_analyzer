import socketio
from aiohttp import web
from pprint import pprint




# sio = socketio.Client()
sio = socketio.AsyncServer(async_mode='aiohttp', async_handlers=True)
app = web.Application()
sio.attach(app)


@sio.event
async def connect(sid, environ):
  pprint(sid)
  print('connection established')
  await sio.emit('conn_test', {'data': 'foobar'})





@sio.on('commodity_data')
async def test_event(sid, data):
  pprint('yes commodity_data worked')
  # pprint(data)
  await sio.emit('com_data_conf')


# @sio.event
# def my_message(data):
#     print('message received with ', data)
#     sio.emit('my response', {'response': 'my response'})

# @sio.on('test')
# def test_event(sid):
#   pprint('yes testing worked')

@sio.event
def disconnect(sid):
    print('disconnected from server')


web.run_app(app, host='localhost',port=3004)