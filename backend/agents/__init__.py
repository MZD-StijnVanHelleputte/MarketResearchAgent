from agents.grounding_agent import GroundingAgent
from agents.competition_agent import CompetitionAgent
from agents.distributors_agent import DistributorsAgent
from agents.customers_agent import CustomersAgent
from agents.mining_projects_agent import MiningProjectsAgent
from agents.commodities_agent import CommoditiesAgent
from agents.macro_geopolitics_agent import MacroGeopoliticsAgent
from agents.general_search_agent import GeneralSearchAgent

# Maps domain name → agent class for the parallel dispatch controller in core/graph.py.
DOMAIN_AGENTS: dict[str, type] = {
    "competition":       CompetitionAgent,
    "distributors":      DistributorsAgent,
    "customers":         CustomersAgent,
    "mining_projects":   MiningProjectsAgent,
    "commodities":       CommoditiesAgent,
    "macro_geopolitics": MacroGeopoliticsAgent,
    "general_search":    GeneralSearchAgent,
}
