-- RL environment simplification

local ALLOWED_JOKER_LIST ={
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

------------------------------------------------------------------
-- STRIP EDITION from a joker card (foil, holo, polychrome etc)
------------------------------------------------------------------
local function strip_edition(card)
    if not card then return end
    if card.edition then
        card:set_edition(nil, true)
    end
end

------------------------------------------------------------------
-- STRIP ENHANCEMENT from a playing card
------------------------------------------------------------------
local function strip_enhancement(card)
    if not card then return end
    if card.config and card.config.center and card.config.center.set == "Enhanced" then
        card:set_ability(G.P_CENTERS["c_base"])
    end
end

------------------------------------------------------------------
-- STRIP SEAL from a playing card
------------------------------------------------------------------
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

    -- ALWAYS remove disallowed cards every frame
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

    -- Mark filled after 0.5s — no manual card creation
    if shop_filled then return end
    if love.timer.getTime() - shop_enter_time < 0.5 then return end
    shop_filled = true
end
------------------------------------------------------------------
-- HOOK GAME LOOP
------------------------------------------------------------------
local orig_update = Game.update
function Game:update(dt)
    orig_update(self, dt)
    enforce_shop()

    -- Force face-up cards in ALL states (not just blind select)
    if G.hand and G.hand.cards then
        for _, card in ipairs(G.hand.cards) do
            if card.facing and card.facing == "back" then
                card:flip()
            end
        end
    end

    -- Force hand size to always be 8
    if G.GAME and G.GAME.hand_size and G.GAME.hand_size ~= 8 then
        G.GAME.hand_size = 8
        if G.hand then G.hand.config.card_limit = 8 end
    end

    -- Only enforce round-start discard count
    if G.GAME and G.GAME.current_round and G.GAME.round_resets then
        G.GAME.round_resets.discards = 4
    end

    if G.STATE == G.STATES.BLIND_SELECT then
        if G.GAME then
            if G.GAME.tags then G.GAME.tags = {} end
            if G.GAME.blind_on_deck then
                local boss = G.P_BLINDS[G.GAME.blind_on_deck]
                if boss and boss.boss then boss.effect = "" end
            end
        end
        if G.P_BLINDS then
            for _, blind in pairs(G.P_BLINDS) do
                blind.tag_effect = ""
                blind.tag = nil
            end
        end
    end
end

-- Disable skip entirely by overriding the function
local orig_skip_blind = G.FUNCS.skip_blind
G.FUNCS.skip_blind = function(e)
    -- do nothing — skipping disabled
end

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
            if not owned[key] and not in_shop[key] then
                table.insert(available, key)
            end
        end

        if #available > 0 then
            local replace_key = available[math.random(#available)]
            card:set_ability(G.P_CENTERS[replace_key])
            strip_edition(card)
        end
        -- If no available jokers, just return card as-is
        -- enforce_shop will remove it cleanly next frame
    end
    return card  -- NEVER return nil
end

------------------------------------------------------------------
-- HOOK START RUN
------------------------------------------------------------------
local orig_start_run = Game.start_run
function Game:start_run(...)
    -- Strip boss effects BEFORE orig_start_run so get_new_boss still works
    -- but effects are already neutralized
    if G.P_BLINDS then
        for _, blind in pairs(G.P_BLINDS) do
            if blind.boss then
                -- Only null the specific effect-triggering fields
                blind.boss.triggered = false
                if type(blind.boss) == "table" then
                    blind.boss.effect = nil
                    blind.boss.vars = {}
                end
            end
            blind.effect = ""
            blind.vars = {}
            blind.pre_boss = nil
        end
    end

    orig_start_run(self, ...)

    -- Lock hand_size to never go below 8
    if G.GAME then
        local mt = getmetatable(G.GAME) or {}
        local existing_newindex = mt.__newindex
        mt.__newindex = function(t, k, v)
            if k == "hand_size" then
                rawset(t, k, math.max(v, 8))
                if G.hand then G.hand.config.card_limit = math.max(v, 8) end
            elseif existing_newindex then
                existing_newindex(t, k, v)
            else
                rawset(t, k, v)
            end
        end
        setmetatable(G.GAME, mt)
    end

    G.E_MANAGER:add_event(Event({
        trigger = "condition",
        blocking = false,
        func = function()
            if not G.deck or not G.deck.cards then return false end
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


    local orig_set_blind = Blind.set_blind

        function Blind:set_blind(blind, reset, silent)

        local target_hand_size = 8
        local target_discards = 4
        local target_hands = 4

        orig_set_blind(self, blind, reset, silent)

        --------------------------------------------------
        -- REMOVE BOSS CHIP MODIFIERS (The Wall etc.)
        --------------------------------------------------
        if self and G.GAME and G.GAME.blind then
            local normal_chips = get_blind_amount(G.GAME.round_resets.ante)

            self.chips = normal_chips
            G.GAME.blind.chips = normal_chips
        end

        --------------------------------------------------
        -- RESTORE RL CONSTANTS
        --------------------------------------------------
        if G.GAME then

            G.GAME.hand_size = target_hand_size

            if G.hand then
                G.hand.config.card_limit = target_hand_size
            end

            if G.GAME.round_resets then
                G.GAME.round_resets.discards = target_discards
                G.GAME.round_resets.hands = target_hands
            end

            if G.GAME.current_round then
                G.GAME.current_round.discards_left =
                    math.max(G.GAME.current_round.discards_left or 0, target_discards)

                G.GAME.current_round.hands_left =
                    math.max(G.GAME.current_round.hands_left or 0, target_hands)
            end
        end
    end
end


sendInfoMessage("RL SIMPLIFY LOADED", "BB.RL")