from model.base import CovidModel, get_parameters
from utils import Propaganda

def test_propaganda():
    model = CovidModel()
    get_parameters().params['risk_tolerance_mean'] = 1.0
    propaganda = Propaganda(model, 5)
    model.add_listener(propaganda)
    assert get_parameters().params['risk_tolerance_mean'] == 1.0
    for i in range(5):
        model.step()
        assert get_parameters().params['risk_tolerance_mean'] == 1.0
    model.step()
    assert get_parameters().params['risk_tolerance_mean'] == 0.9
    
    model.step()
    assert get_parameters().params['risk_tolerance_mean'] == 0.9
    model.step()
    assert get_parameters().params['risk_tolerance_mean'] == 0.9
    model.step()
    assert get_parameters().params['risk_tolerance_mean'] == 0.8

    for i in range(2): model.step()
    assert get_parameters().params['risk_tolerance_mean'] == 0.8

    for i in range(7):
        for j in range(3):
            model.step()
            assert (get_parameters().params['risk_tolerance_mean'] - (0.7 - (0.1 * i))) < 0.001

    for i in range(100): 
        model.step()
        assert get_parameters().params['risk_tolerance_mean'] == 0.1
