#!/usr/bin/env python3

from typing import Iterable, Optional, Dict, Tuple
from vacuumworld import run
from vacuumworld.model.actions.vwactions import VWAction
from vacuumworld.model.actions.vwmove_action import VWMoveAction
from vacuumworld.model.actions.vwturn_action import VWTurnAction
from vacuumworld.model.actions.vwclean_action import VWCleanAction
from vacuumworld.model.actions.vwidle_action import VWIdleAction
from vacuumworld.model.actor.mind.surrogate.vw_llm_actor_mind_surrogate import VWLLMActorMindSurrogate
from vacuumworld.common.vwdirection import VWDirection
from vacuumworld.common.vworientation import VWOrientation
from vacuumworld.model.actions.vwbroadcast_action import VWBroadcastAction
from vacuumworld.common.vwcolour import VWColour
from google.genai.types import GenerateContentResponse


# ----------------------------
# WHITE AGENT
# ----------------------------
class WhiteLLMMind(VWLLMActorMindSurrogate):
    def __init__(self) -> None:
        super().__init__(dot_env_path=".env")
        self.known_width: Optional[int] = None
        self.known_height: Optional[int] = None
        self.observed: set[Tuple[int,int]] = set()
        self.dirt_map: Dict[Tuple[int,int], str] = {}
        self.phase: str = "find_width"
        self.last_row_direction: str = "WEST"
        self.last_visited: Optional[Tuple[int,int]] = None
        self.map_broadcasted: bool = False
        self.cleaned: set[Tuple[int,int]] = set()
        self.moving_up_row = False
        self.next_row_direction = self.last_row_direction
        self.prev_phase: Optional[str] = None
        self.just_blocked_turn = False

        self.colour_name = "white"

        self.visited: set[Tuple[int, int]] = set()

        # Fallback: track last positions to detect repeated MOVE_FORWARD that doesn’t move
        self.last_actions: list[Tuple[str, Tuple[int,int]]] = []

    def minimal_turn_action(
    self,
    current: VWOrientation,
    desired: VWOrientation,
    forward_blocked: bool,
    left_free: bool,
    right_free: bool
) -> VWAction:
        """
        Compute the minimal turn action (or MOVE_FORWARD) to face the desired orientation.
        Chooses shortest rotation direction (left/right) and handles 180° correctly.
        """
        directions = [VWOrientation.north, VWOrientation.east, VWOrientation.south, VWOrientation.west]
        current_idx = directions.index(current)
        desired_idx = directions.index(desired)

        diff = (desired_idx - current_idx) % 4

        if diff == 0:
            # Already facing desired
            if not forward_blocked:
                return VWMoveAction()
            # Forward blocked → try left/right
            if left_free:
                return VWTurnAction(VWDirection.left)
            elif right_free:
                return VWTurnAction(VWDirection.right)
            else:
                return VWIdleAction()
        elif diff == 1:
            return VWTurnAction(VWDirection.right)
        elif diff == 2:
            # 180° turn: choose the direction with free adjacent if possible
            if left_free:
                return VWTurnAction(VWDirection.left)
            elif right_free:
                return VWTurnAction(VWDirection.right)
            else:
                return VWTurnAction(VWDirection.right)  # arbitrary if blocked both sides
        elif diff == 3:
            return VWTurnAction(VWDirection.left)

    def revise(self) -> None:
        try:
            pos = self.get_own_position()
            x, y = int(pos.get_x()), int(pos.get_y())
            self.last_visited = (x, y)
            self.visited.add((x,y))
            orient = self.get_own_orientation()
            obs = self.get_latest_observation()

            # Record observed tiles and dirt
            for getter in [obs.get_center, obs.get_forward, obs.get_left,
                           obs.get_right, obs.get_forwardleft, obs.get_forwardright]:
                opt_loc = getter()
                if not opt_loc.is_empty():
                    loc = opt_loc.or_else_raise()
                    cpos = (int(loc.get_coord().get_x()), int(loc.get_coord().get_y()))
                    self.observed.add(cpos)
                    if loc.has_dirt():
                        colour = str(loc.get_dirt_appearance().or_else_raise().get_colour())
                        self.dirt_map[cpos] = colour

            # Width detection
            if self.phase == "find_width" and orient == VWOrientation.east and obs.is_wall_immediately_ahead():
                self.known_width = x + 1
                self.phase = "find_height"
                print(f"[WHITE] Width found: {self.known_width}")

            # Height detection
            elif self.phase == "find_height" and orient == VWOrientation.south and obs.is_wall_immediately_ahead():
                self.known_height = y + 1
                print(f"[WHITE] Height found: {self.known_height}")
                self.phase = "zigzag"

            # Broadcast if full map observed
            if self.phase == "zigzag" and len(self.observed) == self.known_width * self.known_height \
               and not self.map_broadcasted:
                print("[WHITE] Entire map observed! Preparing to broadcast...")
                self.phase = "broadcasting"

        except Exception as e:
            print(f"[WHITE] revise error: {e}")

    def decide(self) -> Iterable[VWAction]:
        try:
            pos = self.get_own_position()
            x, y = int(pos.get_x()), int(pos.get_y())
            orient = self.get_own_orientation()
            obs = self.get_latest_observation()

            print(f"[WHITE DEBUG] Phase={self.phase}, pos=({x},{y}), orient={orient.name}, last_row_dir={self.last_row_direction}")

            # -------------------------
            # BLOCKED PHASE
            # -------------------------
            forward_loc = obs.get_forward()
            left_loc = obs.get_left()
            right_loc = obs.get_right()

            ahead_has_actor = (
                not forward_loc.is_empty() and
                forward_loc.or_else_raise().has_actor()
            )

            # Left cell
            left_blocked = True
            unvisited_left = False
            if not left_loc.is_empty():
                left_cell = left_loc.or_else_raise()
                lx, ly = int(left_cell.get_coord().get_x()), int(left_cell.get_coord().get_y())
                if self.known_width is not None and self.known_height is not None:
                    left_blocked = (
                        left_cell.has_actor() or
                        lx < 0 or lx >= self.known_width or
                        ly < 0 or ly >= self.known_height
                    )
                else:
                    left_blocked = left_cell.has_actor()
                unvisited_left = (lx, ly) not in self.visited
            else:
                left_blocked = False
                unvisited_left = True

            # Right cell
            right_blocked = True
            unvisited_right = False
            if not right_loc.is_empty():
                right_cell = right_loc.or_else_raise()
                rx, ry = int(right_cell.get_coord().get_x()), int(right_cell.get_coord().get_y())
                if self.known_width is not None and self.known_height is not None:
                    right_blocked = (
                        right_cell.has_actor() or
                        rx < 0 or rx >= self.known_width or
                        ry < 0 or ry >= self.known_height
                    )
                else:
                    right_blocked = right_cell.has_actor()
                unvisited_right = (rx, ry) not in self.visited
            else:
                right_blocked = False
                unvisited_right = True

            print(f"Ahead: {'blocked' if ahead_has_actor else 'free'}\n"
                f"Left: {'blocked' if left_blocked else 'free'}, Unvisited: {unvisited_left}\n"
                f"Right: {'blocked' if right_blocked else 'free'}, Unvisited: {unvisited_right}")

            # Enter blocked phase if forward blocked
            if self.phase == "blocked" or ahead_has_actor:
                if self.phase != "blocked":
                    self.prev_phase = self.phase
                self.phase = "blocked"

                # Forward now clear → return to previous phase
                if not ahead_has_actor:
                    self.phase = self.prev_phase
                    return [VWMoveAction()]

                # LLM prompt for blocked situation
                prompt = f"""
                You are blocked by an agent directly ahead.

                Ahead: {'blocked' if ahead_has_actor else 'free'}
                Left: {'blocked' if left_blocked else 'free'}
                Right: {'blocked' if right_blocked else 'free'}

                Flags:
                - Left leads to an unvisited cell: {unvisited_left}
                - Right leads to an unvisited cell: {unvisited_right}

                Rules (FOLLOW EXACTLY):
                1. IF right cell is FREE and UNVISITED then TURN_RIGHT
                2. If left cell is FREE and UNVISITED then TURN_LEFT
                3. If neither left nor right lead to unvisited cells, but one is free → TURN towards the free cell.
                4. If all blocked → TURN_LEFT
                5. Do not move forward while an agent is ahead.

                Output EXACTLY ONE action: MOVE_FORWARD, TURN_LEFT, or TURN_RIGHT
                Do NOT include text, punctuation, or explanation.
                """
                response = self.decide_physical_with_ai(prompt)
                action = self.parse_gemini_response(response)
                print(f"[WHITE BLOCKED] LLM suggested action: {action}")


                if isinstance(action, VWTurnAction):
                    self.just_blocked_turn = True
                    return [action]

                if getattr(self, "just_blocked_turn", False):
                    self.just_blocked_turn = False
                    if not ahead_has_actor:
                        return [VWMoveAction()]
                    return [VWIdleAction()]

                return [action]

            # -------------------------
            # WIDTH / HEIGHT PHASE
            # -------------------------
            if self.phase == "find_width":
                if orient != VWOrientation.east:
                    return [VWTurnAction(VWDirection.right)]
                forward_loc = obs.get_forward()
                ahead_has_actor = not forward_loc.is_empty() and forward_loc.or_else_raise().has_actor()
                if ahead_has_actor:
                    self.prev_phase = "find_width"
                    self.phase = "blocked"
                    # The blocked-phase logic at the top of decide() will handle this
                    return [VWIdleAction()]
                if not obs.is_wall_immediately_ahead():
                    return [VWMoveAction()]
                return [VWIdleAction()]

            if self.phase == "find_height":
                if orient != VWOrientation.south:
                    return [VWTurnAction(VWDirection.right)]
                forward_loc = obs.get_forward()
                ahead_has_actor = not forward_loc.is_empty() and forward_loc.or_else_raise().has_actor()
                if ahead_has_actor:
                    self.prev_phase = "find_height"
                    self.phase = "blocked"
                    return [VWIdleAction()]
                if obs.is_wall_immediately_ahead():
                    self.known_height = y + 1
                    self.phase = "zigzag"
                return [VWMoveAction()]

            # -------------------------
            # ZIGZAG PHASE (working version)
            # -------------------------
            if self.phase == "zigzag":
                print("[WHITE ZIGZAG] Starting zigzag logic")

                # Determine if forward move is blocked by wall or actor
                fwd = obs.get_forward()
                left_loc = obs.get_left()
                right_loc = obs.get_right()

                # Compute ahead_blocked including walls for LLM
                forward_blocked_by_wall = obs.is_wall_immediately_ahead()
                forward_blocked_by_actor = not fwd.is_empty() and fwd.or_else_raise().has_actor()
                ahead_blocked = forward_blocked_by_wall or forward_blocked_by_actor \
                                or (x == 0 and orient == VWOrientation.west) \
                                or (x == self.known_width-1 and orient == VWOrientation.east) \
                                or (y == 0 and orient == VWOrientation.north) \
                                or (y == self.known_height-1 and orient == VWOrientation.south)

                left_blocked = not left_loc.is_empty() and left_loc.or_else_raise().has_actor()
                right_blocked = not right_loc.is_empty() and right_loc.or_else_raise().has_actor()
                # Row end detection
                at_row_end = (x == 0 and self.last_row_direction == "WEST") or \
                            (x == self.known_width-1 and self.last_row_direction == "EAST")

                # Pre-turn to north if at row end
                if at_row_end and not self.moving_up_row:
                    self.next_row_direction = "EAST" if self.last_row_direction == "WEST" else "WEST"
                    if orient != VWOrientation.north:
                        print("[WHITE ZIGZAG] At row end → pre-turn to north")
                        return [self.minimal_turn_action(
                            orient,
                            VWOrientation.north,
                            forward_blocked=True,
                            left_free=not left_blocked,
                            right_free=not right_blocked
                        )]
                    else:
                        print("[WHITE ZIGZAG] At row end → start moving up")
                        self.moving_up_row = True

                # Move forward if moving up
                if self.moving_up_row:
                    self.moving_up_row = False
                    self.last_row_direction = self.next_row_direction
                    return [VWMoveAction()]

                # LLM prompt for zigzag move (enhanced instructions added)
                prompt = f"""
You are controlling a vacuum agent in a {self.known_width}x{self.known_height} grid.

Current position: ({x},{y})
Current orientation: {orient.name}
Last row direction: {self.last_row_direction}

Grid dimensions: width={self.known_width}, height={self.known_height}
Visited cells: {','.join([f'({vx},{vy})' for vx, vy in self.observed]) if self.observed else 'none'}
Observed dirt locations: {','.join([f'({dx},{dy})' for dx, dy in self.dirt_map.keys()]) if self.dirt_map else 'none'}

Adjacent squares (walls + other agents):
- Ahead: {'blocked' if ahead_blocked else 'free'}
- Left: {'blocked' if left_blocked else 'free'}
- Right: {'blocked' if right_blocked else 'free'}

Flags:
- At row end (for zigzag): {at_row_end}
- Currently moving up a row: {self.moving_up_row}
- Next row direction after moving up: {self.next_row_direction}

Additional instructions:
- If the square in your last row direction is blocked by a wall, first attempt to move north.
- If north is also blocked (by wall or actor), choose the next free direction clockwise (east → south → west).
- Always avoid moving into walls or other agents.
- Prioritize moving into unvisited cells if available.

Rules for zigzag (FOLLOW EXACTLY):
1. IF Currently moving up a row = True:
- Output ONLY MOVE_FORWARD to move north one cell.
2. ELSE IF At row end = True:
- Begin moving up to the next row (set Currently moving up a row = True).
- Output the action needed to move one cell north.
3. ELSE:
- Move forward if the square ahead is unvisited and free.
- TURN_LEFT or TURN_RIGHT only if needed to avoid walls or continue zigzag in last_row_direction.
4. Avoid already visited cells unless no unvisited square is reachable.
5. Avoid squares occupied by other agents.
6. Do not move outside the grid.
7. After moving up, TURN to face Next row direction to continue zigzag.

Output:
- Provide EXACTLY ONE action: MOVE_FORWARD, TURN_LEFT, or TURN_RIGHT.
- DO NOT include explanations, punctuation, code, or extra text.
- Output only the move name.
"""
                response = self.decide_physical_with_ai(prompt)
                llm_action = self.parse_gemini_response(response)
                print(f"[WHITE ZIGZAG] LLM suggested action: {llm_action}")

                # Blocked fallback
                if isinstance(llm_action, VWMoveAction) and ahead_blocked:
                    self.prev_phase = "zigzag"
                    self.phase = "blocked"
                    return [VWIdleAction()]

                # Default desired orientation along current row
                desired_orientation = VWOrientation.west if self.last_row_direction == "WEST" else VWOrientation.east

                # Use minimal_turn_action to decide final action (move or turn)
                action_needed = self.minimal_turn_action(
                    orient,
                    desired_orientation,
                    forward_blocked=ahead_blocked,
                    left_free=not left_blocked,
                    right_free=not right_blocked
                )
                return [action_needed]

            # -------------------------
            # BROADCASTING PHASE
            # -------------------------
            if self.phase == "broadcasting":
                dirt_list = [{"x": dx, "y": dy, "colour": colour} for (dx, dy), colour in self.dirt_map.items()]
                self.map_broadcasted = True
                self.phase = "cleaning"
                return [VWBroadcastAction(message={"dirt": dirt_list}, sender_id=self.get_own_id())]

            # -------------------------
            # CLEANING PHASE
            # -------------------------
            # -------------------------
            # CLEANING PHASE (LLM-BASED for White Agent)
            # -------------------------
            if self.phase == "cleaning":
                center = obs.get_center()

                # Check if standing on dirt that needs cleaning
                standing_on_dirt = False
                dirt_colour_here = None
                current_pos = (x, y)

                if not center.is_empty():
                    c = center.or_else_raise()
                    if c.has_dirt():
                        dirt_app = c.get_dirt_appearance().or_else_raise()
                        dirt_colour_here = dirt_app.get_colour().name.lower()
                        standing_on_dirt = (current_pos not in self.cleaned)

                # If standing on any dirt, clean it
                if standing_on_dirt:
                    print(f"[WHITE] Cleaning {dirt_colour_here} dirt at {current_pos}")
                    self.cleaned.add(current_pos)
                    return [VWCleanAction()]

                # Calculate remaining dirt targets (white cleans ALL dirt)
                remaining_dirt = [pos for pos in self.dirt_map.keys() if pos not in self.cleaned]

                # If no dirt left, idle
                if not remaining_dirt:
                    print("[WHITE] All dirt cleaned, idling")
                    return [VWIdleAction()]

                # Find nearest dirt target
                target = min(remaining_dirt, key=lambda t: abs(t[0]-x) + abs(t[1]-y))
                tx, ty = target
                manhattan_distance = abs(tx - x) + abs(ty - y)
                target_colour = self.dirt_map.get(target, "unknown")

                # Calculate direction to target
                dx, dy = tx - x, ty - y

                # Determine what orientation would move us closer
                if dx != 0:
                    desired_orientation = VWOrientation.east if dx > 0 else VWOrientation.west
                    direction_name = "east" if dx > 0 else "west"
                elif dy != 0:
                    desired_orientation = VWOrientation.south if dy > 0 else VWOrientation.north
                    direction_name = "south" if dy > 0 else "north"
                else:
                    desired_orientation = orient
                    direction_name = orient.name

                # Get adjacent cell information
                fwd = obs.get_forward()
                left_loc = obs.get_left()
                right_loc = obs.get_right()

                # Check forward
                forward_blocked = obs.is_wall_immediately_ahead()
                forward_has_actor = False
                forward_has_dirt = False
                forward_dirt_colour = None
                forward_pos = None

                if not fwd.is_empty():
                    fwd_cell = fwd.or_else_raise()
                    forward_has_actor = fwd_cell.has_actor()
                    forward_blocked = forward_blocked or forward_has_actor
                    forward_pos = (int(fwd_cell.get_coord().get_x()), int(fwd_cell.get_coord().get_y()))

                    if fwd_cell.has_dirt():
                        forward_has_dirt = True
                        forward_dirt_colour = fwd_cell.get_dirt_appearance().or_else_raise().get_colour().name.lower()

                # Check left
                left_blocked = False
                left_has_actor = False
                left_has_dirt = False
                left_dirt_colour = None
                left_pos = None

                if not left_loc.is_empty():
                    left_cell = left_loc.or_else_raise()
                    left_has_actor = left_cell.has_actor()
                    left_blocked = left_has_actor
                    left_pos = (int(left_cell.get_coord().get_x()), int(left_cell.get_coord().get_y()))

                    if left_cell.has_dirt():
                        left_has_dirt = True
                        left_dirt_colour = left_cell.get_dirt_appearance().or_else_raise().get_colour().name.lower()

                # Check right
                right_blocked = False
                right_has_actor = False
                right_has_dirt = False
                right_dirt_colour = None
                right_pos = None

                if not right_loc.is_empty():
                    right_cell = right_loc.or_else_raise()
                    right_has_actor = right_cell.has_actor()
                    right_blocked = right_has_actor
                    right_pos = (int(right_cell.get_coord().get_x()), int(right_cell.get_coord().get_y()))

                    if right_cell.has_dirt():
                        right_has_dirt = True
                        right_dirt_colour = right_cell.get_dirt_appearance().or_else_raise().get_colour().name.lower()

                # Count dirt by colour
                orange_count = sum(1 for pos, col in self.dirt_map.items() if col.lower() == "orange" and pos not in self.cleaned)
                green_count = sum(1 for pos, col in self.dirt_map.items() if col.lower() == "green" and pos not in self.cleaned)

                # Build list of remaining dirt
                dirt_list_str = ', '.join([f"({d[0]},{d[1]}):{self.dirt_map[d]}" for d in remaining_dirt[:5]])
                if len(remaining_dirt) > 5:
                    dirt_list_str += f" ... +{len(remaining_dirt)-5} more"

                # Build comprehensive LLM prompt
                prompt = f"""You are the WHITE cleaning agent in a {self.known_width}x{self.known_height} grid.

MISSION: Navigate to and clean ALL remaining dirt (any colour) in the grid.

CURRENT STATE:
- Your position: ({x}, {y})
- Your orientation: {orient.name}
- Nearest dirt target: ({tx}, {ty}) - {target_colour} colour
- Manhattan distance to target: {manhattan_distance}
- Direction to target: {direction_name}
- Desired orientation to reach target: {desired_orientation.name}

REMAINING DIRT STATUS:
- Total dirt remaining: {len(remaining_dirt)}
  - Orange dirt: {orange_count}
  - Green dirt: {green_count}
- All remaining dirt: {dirt_list_str}

CLEANING PROGRESS:
- Total cleaned: {len(self.cleaned)}
- Last 3 cleaned positions: {list(self.cleaned)[-3:] if self.cleaned else 'none'}

ADJACENT CELLS:
Forward (ahead):
  - Position: {forward_pos if forward_pos else 'wall/unknown'}
  - Blocked by wall: {obs.is_wall_immediately_ahead()}
  - Blocked by actor: {forward_has_actor}
  - Overall blocked: {forward_blocked}
  - Has dirt: {forward_has_dirt}
  - Dirt colour: {forward_dirt_colour if forward_has_dirt else 'none'}

Left:
  - Position: {left_pos if left_pos else 'wall/unknown'}
  - Blocked by actor: {left_has_actor}
  - Overall blocked: {left_blocked}
  - Has dirt: {left_has_dirt}
  - Dirt colour: {left_dirt_colour if left_has_dirt else 'none'}

Right:
  - Position: {right_pos if right_pos else 'wall/unknown'}
  - Blocked by actor: {right_has_actor}
  - Overall blocked: {right_blocked}
  - Has dirt: {right_has_dirt}
  - Dirt colour: {right_dirt_colour if right_has_dirt else 'none'}

NAVIGATION RULES (FOLLOW EXACTLY):
1. You clean ALL colours of dirt (orange, green, or any other).

2. If standing on dirt: (Handled automatically - you won't see this)

3. Priority navigation:
   - If adjacent cell has dirt: Turn to face it, then move to it
   - Otherwise: Navigate to nearest dirt at ({tx}, {ty})

4. Movement logic:
   - If facing desired orientation ({desired_orientation.name}) AND forward is FREE: MOVE_FORWARD
   - If facing desired orientation BUT forward is BLOCKED: Turn to find alternate path
   - If NOT facing desired orientation: Turn towards it

5. When blocked, choose turn direction based on:
   - Which direction has dirt visible
   - Which direction is free and moves closer to target
   - If both blocked, turn towards less-blocked side

6. CRITICAL: NEVER output MOVE_FORWARD when forward is blocked (wall or other agent).

7. If surrounded by orange/green agents cleaning their dirt:
   - Be patient, turn to face free directions
   - Wait for them to move, then proceed to your target

DECISION STRATEGY:
Step 1: Check if adjacent cells have dirt → prioritize moving to them
Step 2: Check alignment with target orientation
Step 3: Check if forward is clear
Step 4: Decide: MOVE_FORWARD (if aligned & clear) OR TURN (if blocked or misaligned)

OUTPUT ONE ACTION ONLY:
- MOVE_FORWARD (only if forward is clear and brings you closer to dirt)
- TURN_LEFT (to avoid obstacles or reorient towards target)
- TURN_RIGHT (to avoid obstacles or reorient towards target)

Output EXACTLY ONE action name. NO explanations, punctuation, or extra text.
"""

                print(f"[WHITE CLEANING] Calling LLM...")
                print(f"  Position: ({x},{y}), Target: ({tx},{ty}), Distance: {manhattan_distance}")
                print(f"  Orient: {orient.name}, Desired: {desired_orientation.name}")
                print(f"  Forward: {'blocked' if forward_blocked else 'free'}, Left: {'blocked' if left_blocked else 'free'}, Right: {'blocked' if right_blocked else 'free'}")

                # Call LLM
                response = self.decide_physical_with_ai(prompt)
                action = self.parse_gemini_response(response)

                print(f"[WHITE CLEANING] LLM suggested: {action.__class__.__name__}")

                # Safety check: if LLM suggests moving forward but path is blocked, override
                if isinstance(action, VWMoveAction) and forward_blocked:
                    print(f"[WHITE CLEANING] Override: Forward blocked, turning instead")
                    # Use minimal_turn_action as fallback
                    action = self.minimal_turn_action(
                        orient,
                        desired_orientation,
                        forward_blocked=True,
                        left_free=not left_blocked,
                        right_free=not right_blocked
                    )

                return [action]

        except Exception as e:
            print(f"[WHITE] decide error: {e}")
            return [VWIdleAction()]


    def parse_gemini_response(self, response: GenerateContentResponse) -> VWAction:
        try:
            if isinstance(response, VWAction):
                return response
            candidates = getattr(response, "candidates", None)
            content = candidates[0].content.parts[0].text if candidates else response
            move = str(content).strip().upper()
            if move == "MOVE_FORWARD":
                return VWMoveAction()
            elif move == "TURN_LEFT":
                return VWTurnAction(VWDirection.left)
            elif move == "TURN_RIGHT":
                return VWTurnAction(VWDirection.right)
            else:
                return VWIdleAction()
        except Exception:
            return VWIdleAction()

    def backup_decide_after_llm_error(self, original_prompt, error, action_superclass):
        return VWIdleAction()


# ----------------------------
# ORANGE / GREEN AGENTS
# ----------------------------
from typing import Iterable, Dict, Tuple, Optional, List, Set
from vacuumworld.model.actions.vwactions import VWAction
from vacuumworld.model.actions.vwmove_action import VWMoveAction
from vacuumworld.model.actions.vwturn_action import VWTurnAction
from vacuumworld.model.actions.vwclean_action import VWCleanAction
from vacuumworld.model.actions.vwidle_action import VWIdleAction
from vacuumworld.model.actions.vwbroadcast_action import VWBroadcastAction
from vacuumworld.model.actor.mind.surrogate.vw_llm_actor_mind_surrogate import VWLLMActorMindSurrogate
from vacuumworld.common.vwdirection import VWDirection
from vacuumworld.common.vworientation import VWOrientation
from google.genai.types import GenerateContentResponse

class BaseCleanerMind(VWLLMActorMindSurrogate):
    def __init__(self, colour_name: str) -> None:
        super().__init__(dot_env_path=".env")
        self.colour_name: str = colour_name.lower()
        self.map_received: bool = False
        self.dirt_map: Dict[Tuple[int,int], str] = {}
        self.cleaned: Set[Tuple[int,int]] = set()
        self.last_positions: List[Tuple[int,int]] = []
        self.just_blocked_turn: bool = False
        self.prev_phase: Optional[str] = None
        self.phase: str = "normal"

    # ----------------------------
    # Minimal turn action (like White)
    # ----------------------------
    def minimal_turn_action(
        self,
        current: VWOrientation,
        desired: VWOrientation,
        forward_blocked: bool,
        left_free: bool,
        right_free: bool
    ) -> VWAction:
        directions = [VWOrientation.north, VWOrientation.east, VWOrientation.south, VWOrientation.west]
        current_idx = directions.index(current)
        desired_idx = directions.index(desired)
        diff = (desired_idx - current_idx) % 4

        if diff == 0:
            if not forward_blocked:
                return VWMoveAction()
            if left_free:
                return VWTurnAction(VWDirection.left)
            if right_free:
                return VWTurnAction(VWDirection.right)
            return VWIdleAction()
        elif diff == 1:
            return VWTurnAction(VWDirection.right)
        elif diff == 2:
            if left_free:
                return VWTurnAction(VWDirection.left)
            if right_free:
                return VWTurnAction(VWDirection.right)
            return VWTurnAction(VWDirection.right)
        elif diff == 3:
            return VWTurnAction(VWDirection.left)

    # ----------------------------
    # Revise: observe dirt and update map
    # ----------------------------
    def revise(self) -> None:
        try:
            pos = self.get_own_position()
            x, y = int(pos.get_x()), int(pos.get_y())
            obs = self.get_latest_observation()
            self.last_positions.append((x, y))
            if len(self.last_positions) > 4:
                self.last_positions.pop(0)

            # process broadcasted map
            for msg in self.get_latest_received_messages():
                content = msg.get_content()
                if isinstance(content, dict) and "dirt" in content and not self.map_received:
                    self.map_received = True
                    for entry in content["dirt"]:
                        pos_tuple = (int(entry["x"]), int(entry["y"]))
                        colour = entry["colour"].lower()
                        self.dirt_map[pos_tuple] = colour

            # update cleaned if current tile has no dirt
            center = obs.get_center()
            if not center.is_empty():
                c = center.or_else_raise()
                cpos = (int(c.get_coord().get_x()), int(c.get_coord().get_y()))
                if not c.has_dirt():
                    self.cleaned.add(cpos)

        except Exception as e:
            print(f"[{self.colour_name.upper()}] revise error: {e}")

    # ----------------------------
    # Decide method: blocked + cleaning logic
    # ----------------------------
    def decide(self) -> Iterable[VWAction]:
        try:
            # Stay idle until map is received
            if not self.map_received:
                return [VWIdleAction()]

            pos = self.get_own_position()
            x, y = int(pos.get_x()), int(pos.get_y())
            orient = self.get_own_orientation()
            obs = self.get_latest_observation()

            # -------------------------
            # BLOCKED PHASE (if needed)
            # -------------------------
            forward_loc = obs.get_forward()
            left_loc = obs.get_left()
            right_loc = obs.get_right()

            ahead_blocked_by_actor = not forward_loc.is_empty() and forward_loc.or_else_raise().has_actor()
            ahead_blocked_by_wall = obs.is_wall_immediately_ahead()
            ahead_blocked = ahead_blocked_by_actor or ahead_blocked_by_wall

            left_blocked, unvisited_left = True, False
            if not left_loc.is_empty():
                lc = left_loc.or_else_raise()
                lx, ly = int(lc.get_coord().get_x()), int(lc.get_coord().get_y())
                left_blocked = lc.has_actor()
                unvisited_left = (lx, ly) not in self.last_positions
            else:
                left_blocked = False
                unvisited_left = True

            right_blocked, unvisited_right = True, False
            if not right_loc.is_empty():
                rc = right_loc.or_else_raise()
                rx, ry = int(rc.get_coord().get_x()), int(rc.get_coord().get_y())
                right_blocked = rc.has_actor()
                unvisited_right = (rx, ry) not in self.last_positions
            else:
                right_blocked = False
                unvisited_right = True

            # Enter blocked phase if actor or wall ahead
            if self.phase == "blocked" or ahead_blocked:
                if self.phase != "blocked":
                    self.prev_phase = self.phase
                self.phase = "blocked"

                if not ahead_blocked:
                    self.phase = self.prev_phase
                    return [VWMoveAction()]

                prompt = f"""
You are blocked by an agent or wall directly ahead.

Ahead: {'blocked' if ahead_blocked else 'free'}
Left: {'blocked' if left_blocked else 'free'}
Right: {'blocked' if right_blocked else 'free'}

Flags:
- Left leads to an unvisited cell: {unvisited_left}
- Right leads to an unvisited cell: {unvisited_right}

Rules (FOLLOW EXACTLY):
1. IF right cell is FREE and UNVISITED then TURN_RIGHT
2. If left cell is FREE and UNVISITED then TURN_LEFT
3. If neither left nor right lead to unvisited cells, but one is free → TURN towards the free cell.
4. If all blocked → TURN_LEFT
5. Do not move forward while an agent or wall is ahead.

Output EXACTLY ONE action: MOVE_FORWARD, TURN_LEFT, or TURN_RIGHT
Do NOT include text, punctuation, or explanation.
"""
                response = self.decide_physical_with_ai(prompt)
                action = self.parse_gemini_response(response)

                if isinstance(action, VWTurnAction):
                    self.just_blocked_turn = True
                    return [action]

                if getattr(self, "just_blocked_turn", False):
                    self.just_blocked_turn = False
                    fwd = obs.get_forward()
                    if (fwd.is_empty() or not fwd.or_else_raise().has_actor()) and not obs.is_wall_immediately_ahead():
                        return [VWMoveAction()]
                    return [VWIdleAction()]

                return [action]

            # -------------------------
            # CLEANING PHASE (LLM-BASED)
            # -------------------------
            center = obs.get_center()

            # Check if standing on dirt that needs cleaning
            standing_on_dirt = False
            dirt_colour_here = None
            current_pos = (x, y)

            if not center.is_empty():
                c = center.or_else_raise()
                if c.has_dirt():
                    dirt_app = c.get_dirt_appearance().or_else_raise()
                    dirt_colour_here = dirt_app.get_colour().name.lower()
                    # Only clean if it matches our colour and hasn't been cleaned
                    standing_on_dirt = (dirt_colour_here == self.colour_name) and (current_pos not in self.cleaned)

            # If standing on matching dirt, clean immediately
            if standing_on_dirt:
                print(f"[{self.colour_name.upper()}] Cleaning {dirt_colour_here} dirt at {current_pos}")
                self.cleaned.add(current_pos)
                return [VWCleanAction()]

            # Calculate remaining dirt targets (ONLY our colour)
            remaining_dirt = [
                pos for pos, colour in self.dirt_map.items()
                if colour.lower() == self.colour_name and pos not in self.cleaned
            ]

            # If no dirt left of our colour, idle
            if not remaining_dirt:
                print(f"[{self.colour_name.upper()}] No remaining {self.colour_name} dirt, idling")
                return [VWIdleAction()]

            # Find nearest dirt target of our colour
            target = min(remaining_dirt, key=lambda t: abs(t[0]-x) + abs(t[1]-y))
            tx, ty = target
            manhattan_distance = abs(tx - x) + abs(ty - y)

            # Calculate direction to target
            dx, dy = tx - x, ty - y

            # Determine what orientation would move us closer
            if dx != 0:
                desired_orientation = VWOrientation.east if dx > 0 else VWOrientation.west
                direction_name = "east" if dx > 0 else "west"
            elif dy != 0:
                desired_orientation = VWOrientation.south if dy > 0 else VWOrientation.north
                direction_name = "south" if dy > 0 else "north"
            else:
                desired_orientation = orient
                direction_name = orient.name

            # Get adjacent cell information
            fwd = obs.get_forward()
            left_loc = obs.get_left()
            right_loc = obs.get_right()

            # Check forward
            forward_blocked = obs.is_wall_immediately_ahead()
            forward_has_actor = False
            forward_has_dirt = False
            forward_dirt_colour = None
            forward_pos = None
            forward_is_target_colour = False

            if not fwd.is_empty():
                fwd_cell = fwd.or_else_raise()
                forward_has_actor = fwd_cell.has_actor()
                forward_blocked = forward_blocked or forward_has_actor
                forward_pos = (int(fwd_cell.get_coord().get_x()), int(fwd_cell.get_coord().get_y()))

                if fwd_cell.has_dirt():
                    forward_has_dirt = True
                    forward_dirt_colour = fwd_cell.get_dirt_appearance().or_else_raise().get_colour().name.lower()
                    forward_is_target_colour = (forward_dirt_colour == self.colour_name)

            # Check left
            left_blocked = False
            left_has_actor = False
            left_has_dirt = False
            left_dirt_colour = None
            left_pos = None
            left_is_target_colour = False

            if not left_loc.is_empty():
                left_cell = left_loc.or_else_raise()
                left_has_actor = left_cell.has_actor()
                left_blocked = left_has_actor
                left_pos = (int(left_cell.get_coord().get_x()), int(left_cell.get_coord().get_y()))

                if left_cell.has_dirt():
                    left_has_dirt = True
                    left_dirt_colour = left_cell.get_dirt_appearance().or_else_raise().get_colour().name.lower()
                    left_is_target_colour = (left_dirt_colour == self.colour_name)

            # Check right
            right_blocked = False
            right_has_actor = False
            right_has_dirt = False
            right_dirt_colour = None
            right_pos = None
            right_is_target_colour = False

            if not right_loc.is_empty():
                right_cell = right_loc.or_else_raise()
                right_has_actor = right_cell.has_actor()
                right_blocked = right_has_actor
                right_pos = (int(right_cell.get_coord().get_x()), int(right_cell.get_coord().get_y()))

                if right_cell.has_dirt():
                    right_has_dirt = True
                    right_dirt_colour = right_cell.get_dirt_appearance().or_else_raise().get_colour().name.lower()
                    right_is_target_colour = (right_dirt_colour == self.colour_name)

            # Count other colour dirt for context
            other_colour = "orange" if self.colour_name == "green" else "green"
            my_colour_count = len(remaining_dirt)
            other_colour_count = sum(1 for pos, col in self.dirt_map.items()
                                    if col.lower() == other_colour and pos not in self.cleaned)

            # Build list of remaining dirt of our colour
            dirt_list_str = ', '.join([f"({d[0]},{d[1]})" for d in remaining_dirt[:5]])
            if len(remaining_dirt) > 5:
                dirt_list_str += f" ... and {len(remaining_dirt)-5} more"

            # Build comprehensive LLM prompt
            prompt = f"""You are the {self.colour_name.upper()} cleaning agent in a grid world.

MISSION: Navigate to and clean ONLY {self.colour_name.upper()} dirt. You must IGNORE {other_colour} dirt (that's for the {other_colour} agent).

YOUR COLOUR: {self.colour_name.upper()}
OTHER AGENT'S COLOUR: {other_colour.upper()} (IGNORE THIS)

CURRENT STATE:
- Your position: ({x}, {y})
- Your orientation: {orient.name}
- Target {self.colour_name} dirt: ({tx}, {ty})
- Manhattan distance to target: {manhattan_distance}
- Direction to target: {direction_name}
- Desired orientation to reach target: {desired_orientation.name}

REMAINING DIRT STATUS:
- {self.colour_name.upper()} dirt remaining (YOUR JOB): {my_colour_count}
- {other_colour.upper()} dirt remaining (NOT YOUR JOB): {other_colour_count}
- Your remaining {self.colour_name} dirt locations: {dirt_list_str}

CLEANING PROGRESS:
- {self.colour_name.upper()} dirt you've cleaned: {len(self.cleaned)}
- Last 3 cleaned positions: {list(self.cleaned)[-3:] if self.cleaned else 'none'}

ADJACENT CELLS:
Forward (ahead):
  - Position: {forward_pos if forward_pos else 'wall/unknown'}
  - Blocked by wall: {obs.is_wall_immediately_ahead()}
  - Blocked by actor: {forward_has_actor}
  - Overall blocked: {forward_blocked}
  - Has dirt: {forward_has_dirt}
  - Dirt colour: {forward_dirt_colour if forward_has_dirt else 'none'}
  - Is YOUR colour ({self.colour_name}): {forward_is_target_colour}

Left:
  - Position: {left_pos if left_pos else 'wall/unknown'}
  - Blocked by actor: {left_has_actor}
  - Overall blocked: {left_blocked}
  - Has dirt: {left_has_dirt}
  - Dirt colour: {left_dirt_colour if left_has_dirt else 'none'}
  - Is YOUR colour ({self.colour_name}): {left_is_target_colour}

Right:
  - Position: {right_pos if right_pos else 'wall/unknown'}
  - Blocked by actor: {right_has_actor}
  - Overall blocked: {right_blocked}
  - Has dirt: {right_has_dirt}
  - Dirt colour: {right_dirt_colour if right_has_dirt else 'none'}
  - Is YOUR colour ({self.colour_name}): {right_is_target_colour}

NAVIGATION RULES (FOLLOW EXACTLY):
1. CRITICAL: You ONLY clean {self.colour_name.upper()} dirt. DO NOT go towards {other_colour} dirt.

2. If standing on {self.colour_name} dirt:
   - (Handled automatically - you won't see this scenario)

3. Priority navigation:
   - If adjacent cell has {self.colour_name} dirt: Turn to face it, then move to it
   - Otherwise: Navigate to nearest {self.colour_name} dirt at ({tx}, {ty})

4. Movement logic:
   - If facing desired orientation ({desired_orientation.name}) AND forward is FREE: MOVE_FORWARD
   - If facing desired orientation BUT forward is BLOCKED: Turn to find alternate path
   - If NOT facing desired orientation: Turn towards it (TURN_LEFT or TURN_RIGHT)

5. When choosing turn direction:
   - Prefer turning towards cells with {self.colour_name} dirt
   - Prefer turning towards free cells that move closer to target
   - Avoid turning towards {other_colour} dirt (not your job)

6. CRITICAL: NEVER output MOVE_FORWARD when forward is blocked (wall or other agent).

7. If blocked by other agents:
   - Be patient, turn to find alternate path
   - Don't compete with the {other_colour} agent if they're near {other_colour} dirt
   - Wait if necessary by turning or idling

8. Coordination:
   - White agent explores and broadcasts the map first
   - You and {other_colour} agent work in parallel to clean your respective colours
   - Stay out of each other's way when possible

DECISION STRATEGY:
Step 1: Check if adjacent cells have {self.colour_name} dirt → move towards it
Step 2: Check alignment with target orientation ({desired_orientation.name})
Step 3: Check if forward path is clear
Step 4: Decide:
   - MOVE_FORWARD if aligned, clear, and moving towards {self.colour_name} dirt
   - TURN_LEFT or TURN_RIGHT if blocked or need to reorient

OUTPUT ONE ACTION ONLY:
- MOVE_FORWARD (only if forward is clear and brings you closer to {self.colour_name} dirt)
- TURN_LEFT (to avoid obstacles or reorient towards {self.colour_name} target)
- TURN_RIGHT (to avoid obstacles or reorient towards {self.colour_name} target)

Output EXACTLY ONE action name. NO explanations, punctuation, or extra text.
"""

            print(f"[{self.colour_name.upper()} CLEANING] Calling LLM...")
            print(f"  Position: ({x},{y}), Target: ({tx},{ty}), Distance: {manhattan_distance}")
            print(f"  Orient: {orient.name}, Desired: {desired_orientation.name}")
            print(f"  Forward: {'blocked' if forward_blocked else 'free'} (has {self.colour_name} dirt: {forward_is_target_colour})")
            print(f"  Left: {'blocked' if left_blocked else 'free'} (has {self.colour_name} dirt: {left_is_target_colour})")
            print(f"  Right: {'blocked' if right_blocked else 'free'} (has {self.colour_name} dirt: {right_is_target_colour})")

            # Call LLM
            response = self.decide_physical_with_ai(prompt)
            action = self.parse_gemini_response(response)

            print(f"[{self.colour_name.upper()} CLEANING] LLM suggested: {action.__class__.__name__}")

            # Safety check: if LLM suggests moving forward but path is blocked, override
            if isinstance(action, VWMoveAction) and forward_blocked:
                print(f"[{self.colour_name.upper()} CLEANING] Override: Forward blocked, turning instead")
                # Use minimal_turn_action as fallback
                action = self.minimal_turn_action(
                    orient,
                    desired_orientation,
                    forward_blocked=True,
                    left_free=not left_blocked,
                    right_free=not right_blocked
                )

            return [action]

        except Exception as e:
            print(f"[{self.colour_name.upper()}] decide error: {e}")
            return [VWIdleAction()]

    # ----------------------------
    # Gemini response parsing
    # ----------------------------
    def parse_gemini_response(self, response: GenerateContentResponse) -> VWAction:
        try:
            if isinstance(response, VWAction):
                return response
            candidates = getattr(response, "candidates", None)
            content = candidates[0].content.parts[0].text if candidates else response
            move = str(content).strip().upper()
            if move == "MOVE_FORWARD":
                return VWMoveAction()
            elif move == "TURN_LEFT":
                return VWTurnAction(VWDirection.left)
            elif move == "TURN_RIGHT":
                return VWTurnAction(VWDirection.right)
            else:
                return VWIdleAction()
        except Exception:
            return VWIdleAction()

    def backup_decide_after_llm_error(self, original_prompt, error, action_superclass):
        return VWIdleAction()


class OrangeMind(BaseCleanerMind):
    def __init__(self):
        super().__init__("orange")

class GreenMind(BaseCleanerMind):
    def __init__(self):
        super().__init__("green")


# ----------------------------
# Run simulation
# ----------------------------
if __name__ == "__main__":
    run(white_mind=WhiteLLMMind(), orange_mind=OrangeMind(), green_mind=GreenMind())
