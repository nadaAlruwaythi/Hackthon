
import uvicorn
from starlette.staticfiles import StaticFiles
import fastapi.responses
import fastapi_chameleon

from starlette.staticfiles import StaticFiles

from pathlib import Path
from data import db_session
from views import account
from views import home
# from views import test
from views import packages
from views import record
# from views import listRecord
# from views import listagent
app = fastapi.FastAPI()


def main():
    configure(dev_mode=True)
    uvicorn.run(app, host='127.0.0.1', port=9888, debug=True)


def configure(dev_mode: bool):
    configure_templates(dev_mode)
    configure_routes()
    configure_db(dev_mode)


def configure_db(dev_mode: bool):
    file = (Path(__file__).parent / 'db' / 'pypi.sqlite').absolute()
    db_session.global_init(file.as_posix())


def configure_templates(dev_mode: bool):
    fastapi_chameleon.global_init('templates', auto_reload=dev_mode)


def configure_routes():
    app.mount('/static', StaticFiles(directory='static'), name='static')
    app.include_router(home.router)
    app.include_router(record.router)
    # app.include_router(listRecord.router)
    # app.include_router(listagent.router)
    # app.include_router(test.router)
    app.include_router(account.router)
    app.include_router(packages.router)


if __name__ == '__main__':
    main()
else:
    configure(dev_mode=False)

