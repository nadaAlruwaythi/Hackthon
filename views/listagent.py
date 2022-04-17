import fastapi
from fastapi_chameleon import template
from starlette.requests import Request

from viewmodels.home.indexviewmodel import IndexViewModel
from viewmodels.shared.viewmodel import ViewModelBase
from fastapi_chameleon import template
from fastapi import FastAPI ,File, UploadFile ,Request

router = fastapi.APIRouter()


@router.get('/listagent')
@template()
def index(request: Request):
    print('Hi')
    vm = IndexViewModel(request)
    return vm.to_dict()


