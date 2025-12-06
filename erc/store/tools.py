import json
from typing import Literal

import erc3
from pydantic import BaseModel, Field


class Req_ListProducts(BaseModel):
    """
    Retrieves a list of available products.
    USE THIS FIRST to find product names and their SKUs.
    """
    tool: Literal["/products/list"] = "/products/list"
    offset: int = Field(0, description="Pagination offset (skip N items)")
    limit: int = Field(10, description="Number of items to return (max 100)")


class Req_ViewBasket(BaseModel):
    """
    Returns the current contents of the shopping basket and total price.
    """
    tool: Literal["/basket/view"] = "/basket/view"


class Req_AddProductToBasket(BaseModel):
    """
    Adds a specific product to the basket.
    Requires a valid 'sku' obtained from /products/list.
    """
    tool: Literal["/basket/add"] = "/basket/add"
    sku: str = Field(..., description="The unique Product SKU (e.g. 'GPU-4090')")
    quantity: int = Field(..., description="Amount to add")


class Req_RemoveItemFromBasket(BaseModel):
    """
    Removes a product from the basket.
    """
    tool: Literal["/basket/remove"] = "/basket/remove"
    sku: str = Field(..., description="The unique Product SKU to remove")
    quantity: int = Field(0, description="Amount to remove. If 0, removes all items of this SKU.")


class Req_ApplyCoupon(BaseModel):
    """
    Applies a discount coupon code to the basket.
    """
    tool: Literal["/coupon/apply"] = "/coupon/apply"
    coupon: str = Field(..., description="The coupon code (e.g. 'SAVE10')")


class Req_RemoveCoupon(BaseModel):
    """
    Removes the currently applied coupon from the basket.
    """
    tool: Literal["/coupon/remove"] = "/coupon/remove"


class Req_CheckoutBasket(BaseModel):
    """
    Finalizes the purchase. Use this ONLY when the basket contains all required items.
    """
    tool: Literal["/basket/checkout"] = "/basket/checkout"


class ReportTaskCompletion(BaseModel):
    """
    Call this when the task is fully done or if you cannot proceed.
    """
    tool: Literal["report_completion"] = "report_completion"
    final_message: str = Field(..., description="Final answer to the user's task")


ALL_TOOLS = [
    Req_ListProducts,
    Req_ViewBasket,
    Req_AddProductToBasket,
    Req_RemoveItemFromBasket,
    Req_ApplyCoupon,
    Req_RemoveCoupon,
    Req_CheckoutBasket,
    ReportTaskCompletion,
]


def get_tool_signature(model_class):
    """
    Generates a tool description with arguments.
    """
    schema = model_class.model_json_schema()
    tool_name = model_class.model_fields['tool'].default
    props = schema.get("properties", {})
    if "tool" in props:
        del props["tool"]
    args_desc = json.dumps(props)
    return f"Tool: {tool_name}\nDescription: {model_class.__doc__ or 'No description'}\nArguments: {args_desc}"


TOOLS_DESC = "\n\n".join([get_tool_signature(t) for t in ALL_TOOLS])

TOOL_TO_METHOD = {
    "/products/list": "list_products",
    "/basket/view": "view_basket",
    "/basket/add": "add_product_to_basket",
    "/basket/remove": "remove_item_from_basket",
    "/coupon/apply": "apply_coupon",
    "/coupon/remove": "remove_coupon",
    "/basket/checkout": "checkout_basket",
    "report_completion": None
}
