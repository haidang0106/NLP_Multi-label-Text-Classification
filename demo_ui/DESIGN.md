# Design System Strategy: The Intelligent Light

## 1. Overview & Creative North Star
**The Creative North Star: "The Pristine Analyst"**
This design system moves beyond the standard "SaaS Blue" template to create an environment that feels like a high-end, physical workspace. It is built on the philosophy of **Atmospheric Clarity**. Instead of using rigid borders and heavy dividers to organize information, we use light, depth, and tonal shifts to guide the user’s eye. 

The aesthetic is "Editorial Intelligence": a sophisticated blend of generous whitespace, bold indigo accents, and a layered surface architecture that suggests a continuous, fluid digital canvas rather than a collection of disconnected boxes.

---

## 2. Colors & Surface Architecture
The color palette is engineered for high-performance readability and a premium "paper-like" tactile quality.

### The Palette (Material Design Convention)
*   **Primary Core:** `primary` (#24389c) for high-contrast utility and `primary_container` (#3f51b5) for signature brand moments.
*   **Neutral Foundation:** `background` (#f8f9ff) and `surface` (#f8f9ff).
*   **Typography:** `on_surface` (#0d1c2f) provides a deep, slate-like contrast that is easier on the eyes than pure black.

### The "No-Line" Rule
**Standard 1px borders are strictly prohibited for sectioning.** To separate content, use background shifts:
*   Place a `surface_container_low` (#eff4ff) block against the `background` (#f8f9ff).
*   Use `surface_container_highest` (#d5e3fd) to highlight the most critical interactive zones.
*   Visual boundaries must be felt through color temperature, not drawn with lines.

### Glass & Gradient Signature
To elevate the experience, apply a **Linear Gradient** to main Action buttons and Hero elements:
*   *Start:* `primary` (#24389c) | *End:* `primary_container` (#3f51b5).
*   For floating navigation or overlay modals, use **Glassmorphism**: Set the background to `surface` at 80% opacity with a `24px` backdrop-blur. This ensures the UI feels integrated into the environment.

---

## 3. Typography
We utilize **Manrope** for its geometric clarity and modern professional tone. The hierarchy is designed to feel like a financial journal—authoritative yet accessible.

*   **Display (lg/md/sm):** Used for data storytelling. High-impact, 3.5rem to 2.25rem. Keep tracking tight (-0.02em) to maintain a "lockup" feel.
*   **Headline (lg/md/sm):** Editorial anchors. Use `on_surface` (#0d1c2f). Bold weights only.
*   **Body (lg/md/sm):** The workhorse. `body-lg` (1rem) for primary reading; `body-md` (0.875rem) for secondary data.
*   **Labels:** Use `label-md` (0.75rem) in `secondary` (#565c84) for metadata.

---

## 4. Elevation & Depth
Depth in this system is a result of "Tonal Stacking," not artificial shadows.

### The Layering Principle
*   **Level 0 (Floor):** `surface` (#f8f9ff)
*   **Level 1 (Sub-section):** `surface_container_low` (#eff4ff)
*   **Level 2 (Active Component):** `surface_container_lowest` (#ffffff) — This creates a "pop" effect against the slightly darker floor.

### Ambient Shadows
Shadows should only be used for "floating" elements (Modals, Popovers).
*   **Shadow Value:** `0px 12px 32px rgba(13, 28, 47, 0.06)`. 
*   **The "Ghost Border" Fallback:** If a component requires a boundary for accessibility, use `outline_variant` (#c5c5d4) at **15% opacity**.

---

## 5. Components

### Buttons
*   **Primary:** Gradient from `primary` to `primary_container`. Roundedness `full` (9999px) for a high-tech feel.
*   **Secondary:** `surface_container_high` (#dde9ff) background with `on_primary_fixed_variant` (#293ca0) text. No border.
*   **Tertiary:** Text-only in `primary`. Underline on hover only.

### Cards & Containers
*   **Rule:** Forbid divider lines. 
*   **Structure:** Use a `1.5rem` (spacing scale 6) internal padding. Separate headers from content using a shift from `surface_container_low` (header) to `surface_container_lowest` (body).
*   **Corners:** All cards must use `xl` (1.5rem) or `lg` (1rem) radii.

### Input Fields
*   **Static State:** `surface_container_low` background, no border.
*   **Focus State:** 2px solid `primary_container`. Smooth 200ms transition.
*   **Labeling:** Labels should be `label-md`, positioned 0.5rem above the input, never floating inside.

### Signature Component: The "Intelligence Chip"
For AI-driven insights, use a `tertiary_fixed` (#ffdcc6) background with `on_tertiary_fixed` (#301400) text. This warm break from the blue/gray scale signals a "special" or "generated" status.

---

## 6. Do’s and Don’ts

### Do
*   **Do** use asymmetrical layouts. Push a headline to the far left and the content card to the center-right to create "dynamic tension."
*   **Do** use `20` (5rem) or `24` (6rem) spacing for major section gaps to allow the UI to breathe.
*   **Do** use the `primary_fixed` (#dee0ff) color for subtle background highlights behind important icons.

### Don't
*   **Don't** use 100% black (#000000). It breaks the "Atmospheric" softness of the light mode.
*   **Don't** use standard 4px or 8px corners. Stick to the scale: `0.75rem` (md) is the absolute minimum for small elements.
*   **Don't** stack more than three levels of surfaces. If you need more depth, use a subtle ambient shadow, not another color shift.