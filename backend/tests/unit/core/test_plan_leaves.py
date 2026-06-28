"""The plan builder turns a flat entity manifest into a de-overlapped stem→leaf
tree, and re-canonicalizes ticker-scoped tool calls to their owning domain."""
from core.plan_merger import _build_leaves, _recanonicalize_calls
from core.tot.schemas import PlannedToolCall
from services.masterdata_service import MasterDataService


def test_company_in_two_buckets_yields_one_canonical_leaf():
    md = MasterDataService()
    # Caterpillar deliberately mis-placed in the operators bucket as well.
    manifest = {
        "competitors": ["Caterpillar Inc. (CAT)"],
        "operators": ["BHP Group (BHP)", "Caterpillar Inc. (CAT)"],
    }
    leaves = _build_leaves(manifest, md)
    cat_leaves = [lf for lf in leaves if lf.key == "CAT"]
    assert len(cat_leaves) == 1
    assert cat_leaves[0].domain == "competition"
    assert cat_leaves[0].leaf_type == "company"
    # BHP stays a mining operator.
    assert any(lf.domain == "mining_operators" and lf.label == "BHP" for lf in leaves)


def test_leaf_types_and_tools_are_assigned_from_registry():
    md = MasterDataService()
    manifest = {
        "commodities": ["Gold (GC=F)"],
        "regions": ["Chile"],
        "mine_sites": ["Escondida"],
        "demand_side_companies": ["BYD"],
    }
    leaves = {lf.label: lf for lf in _build_leaves(manifest, md)}
    assert leaves["Gold (GC=F)"].leaf_type == "commodity"
    assert leaves["Chile"].leaf_type == "country" and leaves["Chile"].domain == "macroeconomics"
    assert leaves["Escondida"].leaf_type == "mine_site" and leaves["Escondida"].domain == "mining_projects"
    # Demand-side consumers live under general_search as company leaves.
    assert leaves["BYD"].domain == "general_search" and leaves["BYD"].leaf_type == "company"
    # Every leaf carries a non-empty deterministic toolset.
    assert all(lf.tools for lf in leaves.values())


def test_recanonicalize_moves_mistagged_company_call():
    md = MasterDataService()
    calls = [PlannedToolCall(
        tool="get_company_financials", params={"ticker": "CAT"}, domain="mining_operators",
    )]
    out = _recanonicalize_calls(calls, md)
    assert out[0].domain == "competition"
