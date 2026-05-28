-- RL environment simplification

local ALLOWED_JOKER_LIST = {
    "j_joker",
    "j_lusty_joker",
    "j_jolly",
    "j_droll",
    "j_crafty",
    "j_smeared",
    "j_zany",
    "j_mad",
    "j_sly",
    "j_wily",
    "j_abstract",
    "j_half",
    "j_scary_face",
    "j_greedy_joker",
}

local function is_allowed_key(key)
    for _, k in ipairs(ALLOWED_JOKER_LIST) do
        if k == key then return true end
    end
    return false
end

local function is_allowed(card)
    if not card or not card.config then return false end
    return is_allowed_key(card.config.center_key)
end

local function strip_edition(card)
    if not card then return end
    if card.edition then
        card:set_edition(nil, true)
    end
end

local function strip_enhancement(card)
    if not card then return end
    if card.config and card.config.center and card.config.center.set == "Enhanced" then
        card:set_ability(G.P_CENTERS["c_base"])
    end
end

local function strip_seal(card)
    if not card then return end
    if card.seal then
        card:set_seal(nil, true)
    end
end

------------------------------------------------------------------
-- ENFORCE SHOP CONTENTS
------------------------------------------------------------------
local shop_filled = false
local last_joker_count = 0
local shop_enter_time = 0

local function enforce_shop()
    if not G or G.STATE ~= G.STATES.SHOP then
        shop_filled = false
        shop_enter_time = 0
        return
    end

    if not G.shop_jokers or not G.shop_jokers.cards then return end

    local current_joker_count = G.jokers and #G.jokers.cards or 0
    if current_joker_count ~= last_joker_count then
        last_joker_count = current_joker_count
        shop_filled = false
        shop_enter_time = love.timer.getTime()
    end

    if shop_enter_time == 0 then
        shop_enter_time = love.timer.getTime()
    end

    for i = #G.shop_jokers.cards, 1, -1 do
        local card = G.shop_jokers.cards[i]
        if not is_allowed(card) then
            card:remove()
            table.remove(G.shop_jokers.cards, i)
        else
            strip_edition(card)
        end
    end

    if G.shop_booster and G.shop_booster.cards then
        for i = #G.shop_booster.cards, 1, -1 do
            G.shop_booster.cards[i]:remove()
            table.remove(G.shop_booster.cards, i)
        end
    end

    if G.shop_consumables and G.shop_consumables.cards then
        for i = #G.shop_consumables.cards, 1, -1 do
            G.shop_consumables.cards[i]:remove()
            table.remove(G.shop_consumables.cards, i)
        end
    end

    if G.shop_vouchers and G.shop_vouchers.cards then
        for i = #G.shop_vouchers.cards, 1, -1 do
            G.shop_vouchers.cards[i]:remove()
            table.remove(G.shop_vouchers.cards, i)
        end
    end

    if shop_filled then return end
    if love.timer.getTime() - shop_enter_time < 0.5 then return end
    shop_filled = true
end

------------------------------------------------------------------
-- UNIFIED HOOK GAME LOOP
------------------------------------------------------------------
local orig_update = Game.update
function Game:update(dt)
    orig_update(self, dt)

    enforce_shop()

    -- Force all hand cards face up
    if G.hand and G.hand.cards then
        for _, card in ipairs(G.hand.cards) do
            if card.facing and card.facing == "back" then
                card:flip()
            end
        end
    end

    ------------------------------------------------------------------
    -- INDEPENDENT HAND SIZE FORCING (Fixes permanent post-Manacle drops)
    ------------------------------------------------------------------
    if G.GAME and G.GAME.hand_size and G.GAME.hand_size ~= 8 then
        G.GAME.hand_size = 8
    end
    
    if G.hand and G.hand.config and G.hand.config.card_limit ~= 8 then
        G.hand.config.card_limit = 8
    end

    -- Force standard round allocations
    if G.GAME and G.GAME.round_resets then
        G.GAME.round_resets.discards = 4
        G.GAME.round_resets.hands = 4
    end

    ------------------------------------------------------------------
    -- NATIVE BOSS DISABLE (Fixes The Serpent hardcoded draw limit)
    ------------------------------------------------------------------
    if G.GAME and G.GAME.blind then
        G.GAME.blind.disabled = true
    end

    -- Mid-round environment reset/recovery
    if G.GAME and G.GAME.current_round and G.STATE == G.STATES.SELECTING_HAND then
        local cr = G.GAME.current_round

        if cr.hands_played == 0 and cr.discards_used == 0 then
            if cr.hands_left and cr.hands_left < 4 then cr.hands_left = 4 end
            if cr.discards_left and cr.discards_left < 4 then cr.discards_left = 4 end
        end
    end

    if G.STATE == G.STATES.BLIND_SELECT then
        if G.GAME and G.GAME.tags then G.GAME.tags = {} end
        if G.P_BLINDS then
            for _, blind in pairs(G.P_BLINDS) do
                blind.tag_effect = ""
                blind.tag = nil
            end
        end
    end
end

-- Disable skip blind
G.FUNCS.skip_blind = function(e) end

------------------------------------------------------------------
-- INTERCEPT SHOP CARD CREATION
------------------------------------------------------------------
local orig_create_card_for_shop = create_card_for_shop

function create_card_for_shop(area, forced_tag)
    local card = orig_create_card_for_shop(area, nil)

    if card and card.ability and card.ability.set == "Joker" then
        local owned = {}

        if G.jokers and G.jokers.cards then
            for _, c in ipairs(G.jokers.cards) do
                if c.config and c.config.center_key then
                    owned[c.config.center_key] = true
                end
            end
        end

        local in_shop = {}

        if G.shop_jokers and G.shop_jokers.cards then
            for _, c in ipairs(G.shop_jokers.cards) do
                if c.config and c.config.center_key then
                    in_shop[c.config.center_key] = true
                end
            end
        end

        local available = {}

        for _, key in ipairs(ALLOWED_JOKER_LIST) do
            if not owned[key]
            and not in_shop[key]
            and G.P_CENTERS[key] then
                table.insert(available, key)
            end
        end

        if #available > 0 then
            local replace_key = available[math.random(#available)]
            card:set_ability(G.P_CENTERS[replace_key])
            strip_edition(card)
        end
    end

    return card
end

------------------------------------------------------------------
-- HOOK START RUN
------------------------------------------------------------------
local orig_start_run = Game.start_run

function Game:start_run(...)
    orig_start_run(self, ...)

    G.E_MANAGER:add_event(Event({
        trigger = "condition",
        blocking = false,
        func = function()
            if not G.deck or not G.deck.cards then
                return false
            end

            for _, card in pairs(G.deck.cards) do
                if card.config and card.config.card then
                    -- card:change_suit("Hearts")
                    strip_enhancement(card)
                    strip_seal(card)
                end
            end

            return true
        end,
    }))
end

------------------------------------------------------------------
-- DISABLE ALL BOSS BLIND EFFECTS
------------------------------------------------------------------
if Blind then
    function Blind:debuff_hand(cards, poker_hand, handname, check)
        return false
    end

    function Blind:debuff_card(card, from_blind)
        return false
    end

    function Blind:press_play()
    end

    local orig_set_blind = Blind.set_blind
    function Blind:set_blind(blind, reset, silent)
        orig_set_blind(self, blind, reset, silent)
        
        -- Force disable the blind instance natively upon activation
        self.disabled = true

        if self.config and self.config.blind and self.config.blind.name == "The Needle" then
            self.config.blind.hands = 4
        end

        -- Reset boss blind chip requirement to standard 2x (remove Wall)
        if G.GAME.blind and self.boss then
                local base = get_blind_amount(G.GAME.round_resets.ante)
                if base then
                    local standard_chips = math.floor(base * 2)  -- boss is always 2x base
                    self.chips = standard_chips
                    G.GAME.blind.chips = standard_chips
                    if self.chip_text then
                        self.chip_text = number_format(standard_chips)
                    end
                end
            end
    end

    
end

sendInfoMessage("RL SIMPLIFY LOADED", "BB.RL")