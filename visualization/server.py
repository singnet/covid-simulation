from mesa_geo.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter
from model import InfectedModel, SchoolAgent, RestaurantAgent
from mesa_geo.visualization.MapModule import MapModule


class InfectedText(TextElement):
    """
    Display a text count of how many steps have been taken
    """

    def __init__(self):
        pass

    def render(self, model):
        return "Days: " + str(model.steps)


model_params = {
    "show_schools": UserSettableParameter('checkbox', 'Show schools', value=True),
    "show_restaurants": UserSettableParameter('checkbox', 'Show restaurants', value=True),
    "monitored_statistic": UserSettableParameter('choice', 'Monitored statistic', value='infected', choices=['infected']),
}


def infected_draw(agent):
    """
    Portrayal Method for canvas
    """
    portrayal = dict()
    if isinstance(agent, SchoolAgent) or isinstance(agent, RestaurantAgent):
        portrayal["radius"] = agent.map_point_radius
        portrayal["fill"] = True
    portrayal["color"] = agent.map_color
    return portrayal


infected_text = InfectedText()
map_element = MapModule(infected_draw, InfectedModel.MAP_COORDS, 12, 500, 800)
infected_chart = ChartModule(
    [
        {"Label": "infected", "Color": "Blue"},
        #{"Label": "susceptible", "Color": "Green"},
        {"Label": "recovered", "Color": "Magenta"},
        {"Label": "dead", "Color": "Black"},
    ]
)
server = ModularServer(
    InfectedModel,
    [map_element, infected_text, infected_chart],
    "Basic agent-based SIR model",
    model_params,
)
server.launch()
