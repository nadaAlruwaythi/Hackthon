import fastapi
from fastapi_chameleon import template
from starlette.requests import Request


router = fastapi.APIRouter()


@router.get('/test')
@template()
def index(request: Request):
    print('This is Test View')
    vm = IndexViewModel(request)
    return vm.to_dict()



