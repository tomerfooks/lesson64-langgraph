const searchProducts = async ({ query }) => {
  const data = await fetch(
    `https://dummyjson.com/products/search?q=${encodeURIComponent(query)}`,
  );
  const products = await data.json();
  return products;
};

// 2) כלי (Tool): מחפש מוצרים לפי מילת חיפוש, דרך DummyJSON (בלי מפתח API).
const searchProductsTool = tool(searchProducts, {
  name: "search_products",
  description: "מחפש מוצרים בחנות לפי מילת חיפוש ומחזיר שם ומחיר",
  schema: z.object({ query: z.string() }),
});

export default searchProductsTool;
