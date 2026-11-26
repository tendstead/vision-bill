"""Extend the shared line-item tag vocabulary."""

from alembic import op

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO tags (name)
        SELECT seed.name FROM (VALUES
            ('apples'), ('baby_food'), ('bananas'), ('batteries'),
            ('beef'), ('beer'), ('berries'), ('bin_bags'),
            ('biscuits'), ('books'), ('bread'), ('butter'),
            ('cables'), ('canned'), ('car_supplies'), ('carrots'),
            ('cereal'), ('cheese'), ('chips'), ('chocolate'),
            ('citrus'), ('cleaning_products'), ('clothing'), ('cosmetics'),
            ('cream'), ('dairy'), ('deodorant'), ('diapers'),
            ('dish_soap'), ('dishwasher_tablets'), ('eggs'), ('first_aid'),
            ('fish'), ('flour'), ('flowers'), ('frozen'),
            ('fruit'), ('fuel'), ('garden_supplies'), ('gluten_free'),
            ('hardware'), ('ice_cream'), ('juice'), ('kitchenware'),
            ('lactose_free'), ('laundry_detergent'), ('legumes'), ('light_bulbs'),
            ('magazines'), ('meat'), ('medicines'), ('menstrual_products'),
            ('milk'), ('mushrooms'), ('nuts'), ('oats'),
            ('oil'), ('onions'), ('organic'), ('paper_towels'),
            ('pasta'), ('pastries'), ('pet_food'), ('pharmacy'),
            ('plant_milk'), ('plants'), ('pork'), ('potatoes'),
            ('poultry'), ('rice'), ('rolls'), ('salad'),
            ('sauces'), ('sausages'), ('seafood'), ('shampoo'),
            ('shoes'), ('soap'), ('soft_drinks'), ('spices'),
            ('sports_equipment'), ('spreads'), ('stationery'), ('sugar'),
            ('sweets'), ('tea'), ('tickets'), ('tissues'),
            ('tofu'), ('toilet_paper'), ('tomatoes'), ('tools'),
            ('toothpaste'), ('toys'), ('vegan'), ('vegetables'),
            ('vegetarian'), ('water'), ('wine'), ('yogurt')
        ) AS seed(name)
        WHERE NOT EXISTS (SELECT FROM tags WHERE lower(name) = seed.name)
        ON CONFLICT (name) DO NOTHING
    """)


def downgrade() -> None:
    # Tags may predate the upgrade as custom entries or be in use on receipts.
    pass
