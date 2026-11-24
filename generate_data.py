from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(101)
n = 1800
page_views = rng.poisson(7, n).clip(1, 40)
duration = np.clip(rng.gamma(2.2, 3.0, n), .2, 60)
product_views = rng.poisson(3.5, n).clip(0, 25)
cart = rng.poisson(.8, n).clip(0, 8)
returning = rng.binomial(1, .32, n)
device = rng.choice(["mobile", "desktop", "tablet"], n, p=[.62,.32,.06])
source = rng.choice(["search", "social", "direct", "email", "referral"], n)
logit = -4.2 + .045*page_views + .035*duration + .10*product_views + .75*cart + .55*returning
logit += np.where(source=="email", .35, 0) + np.where(device=="desktop", .15, 0)
prob = 1/(1+np.exp(-logit))
purchased = rng.binomial(1, prob)
df = pd.DataFrame({"page_views":page_views, "session_duration_min":duration.round(2),
    "product_views":product_views, "cart_additions":cart, "returning_visitor":returning,
    "device_type":device, "traffic_source":source, "purchased":purchased})
out = Path("data/ecommerce_sessions_demo.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Created {n} synthetic sessions at {out}")
