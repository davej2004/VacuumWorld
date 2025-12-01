#!/usr/bin/env python3

from typing import Iterable, Optional, Set, Tuple, Dict, List

from vacuumworld import run
from vacuumworld.model.actions.vwactions import VWAction
from vacuumworld.model.actions.vwmove_action import VWMoveAction
from vacuumworld.model.actions.vwturn_action import VWTurnAction
from vacuumworld.model.actions.vwclean_action import VWCleanAction
from vacuumworld.model.actions.vwidle_action import VWIdleAction
from vacuumworld.model.actions.vwbroadcast_action import VWBroadcastAction
from vacuumworld.model.actor.mind.surrogate.vwactor_mind_surrogate import VWActorMindSurrogate
from vacuumworld.common.vwdirection import VWDirection
from vacuumworld.common.vworientation import VWOrientation
from vacuumworld.common.vwcolour import VWColour


# ----------------------------
# WhiteMind: perception-aware zigzag + simultaneous cleaning
# ----------------------------
# ----------------------------
# WhiteMind: perception-aware zigzag + simultaneous cleaning
# ----------------------------
class WhiteMind(VWActorMindSurrogate):
    def __init__(self) -> None:
        super().__init__()
        self.known_width: Optional[int] = None
        self.known_height: Optional[int] = None

        self.dirt_map: Dict[Tuple[int, int], str] = {}
        self.visited: Set[Tuple[int, int]] = set()
        self.observed: Set[Tuple[int, int]] = set()  # track all observed cells

        self.phase: str = "find_width"
        self.zigzag_dir: str = "west"

        self.cleaned: Set[Tuple[int, int]] = set()
        self.map_broadcasted: bool = False

        # Flags for actor avoidance
        self.just_turned: bool = False
        self.turn_direction: Optional[VWDirection] = None

    def revise(self) -> None:
        try:
            pos = self.get_own_position()
            x, y = int(pos.get_x()), int(pos.get_y())
            orient = self.get_own_orientation()
            self.visited.add((x, y))
            print(f"[WHITE] Cycle info - Position: ({x},{y}), Orientation: {orient.name}")

            obs = self.get_latest_observation()

            # Update observed squares exploiting perception
            for getter in [obs.get_center, obs.get_forward, obs.get_left,
                           obs.get_right, obs.get_forwardleft, obs.get_forwardright]:
                opt_loc = getter()
                if not opt_loc.is_empty():
                    loc = opt_loc.or_else_raise()
                    cpos = (int(loc.get_coord().get_x()), int(loc.get_coord().get_y()))
                    self.observed.add(cpos)

                    if loc.has_dirt():
                        dirt_app = loc.get_dirt_appearance().or_else_raise()
                        colour = str(dirt_app.get_colour())
                        if cpos not in self.dirt_map:
                            self.dirt_map[cpos] = colour
                            print(f"[WHITE] Found {colour} dirt at {cpos}")

            # Infer width/height
            if self.phase == "find_width" and orient == VWOrientation.east and obs.is_wall_immediately_ahead():
                self.known_width = x + 1
                self.phase = "find_height"
                print(f"[WHITE] Grid width inferred: {self.known_width}")
            elif self.phase == "find_height" and orient == VWOrientation.south and obs.is_wall_immediately_ahead():
                self.known_height = y + 1
                self.phase = "zigzag"
                self.zigzag_dir = "west"
                print(f"[WHITE] Grid height inferred: {self.known_height}")
                print(f"[WHITE] Starting perception-aware zigzag from bottom-right ({self.known_width-1},{self.known_height-1})")

        except Exception as e:
            print(f"[WHITE] revise error: {e}")

    def decide(self) -> Iterable[VWAction]:
        try:
            pos = self.get_own_position()
            x, y = int(pos.get_x()), int(pos.get_y())
            orient = self.get_own_orientation()
            obs = self.get_latest_observation()

            print(f"[WHITE] Decide - Phase: {self.phase}, Position: ({x},{y}), Orientation: {orient.name}")

            # --- Find width ---
            if self.phase == "find_width":
                if orient != VWOrientation.east:
                    print("[WHITE] Turning to face east to find width")
                    return [VWTurnAction(VWDirection.right)]
                if not obs.is_wall_immediately_ahead():
                    print("[WHITE] Moving east to find width")
                    return [VWMoveAction()]
                print("[WHITE] Idle at east wall during width finding")
                return [VWIdleAction()]

            # --- Find height ---
            if self.phase == "find_height":
                if orient != VWOrientation.south:
                    print("[WHITE] Turning to face south to find height")
                    return [VWTurnAction(VWDirection.right)]
                if not obs.is_wall_immediately_ahead():
                    print("[WHITE] Moving south to find height")
                    return [VWMoveAction()]
                print("[WHITE] Idle at south wall during height finding")
                return [VWIdleAction()]

            # --- Perception-aware zigzag ---
            if self.phase == "zigzag":
                print(f"[WHITE] Zigzag direction: {self.zigzag_dir}")

                maxX = self.known_width - 1
                maxY = self.known_height - 1

                unobs = [(i, j) for i in range(self.known_width) for j in range(self.known_height)
                        if (i, j) not in self.observed]

                if not unobs:
                    print("[WHITE] All cells observed, switching to broadcasting")
                    self.phase = "broadcasting"
                    return [VWIdleAction()]

                # Pick nearest unobserved
                target = min(unobs, key=lambda t: abs(t[0]-x)+abs(t[1]-y))
                tx, ty = int(target[0]), int(target[1])
                dx, dy = tx - x, ty - y

                if abs(dx) >= abs(dy) and dx != 0:
                    desired = VWOrientation.east if dx > 0 else VWOrientation.west
                elif dy != 0:
                    desired = VWOrientation.south if dy > 0 else VWOrientation.north
                else:
                    desired = orient

                fwd = obs.get_forward()
                forward_blocked_by_actor = (not fwd.is_empty() and fwd.or_else_raise().has_actor())
                forward_blocked_by_wall = obs.is_wall_immediately_ahead()

                # --- Actor avoidance ---
                if forward_blocked_by_actor and not forward_blocked_by_wall:
                    left_obs = obs.get_left()
                    right_obs = obs.get_right()

                    left_free = False
                    right_free = False

                    if not left_obs.is_empty():
                        left_free = not left_obs.or_else_raise().has_actor()
                    if not right_obs.is_empty():
                        right_free = not right_obs.or_else_raise().has_actor()

                    if left_free:
                        self.just_turned = True
                        self.turn_direction = VWDirection.left
                        print("[WHITE] Actor ahead, turning left to avoid")
                        return [VWTurnAction(VWDirection.left)]
                    elif right_free:
                        self.just_turned = True
                        self.turn_direction = VWDirection.right
                        print("[WHITE] Actor ahead, turning right to avoid")
                        return [VWTurnAction(VWDirection.right)]
                    else:
                        print("[WHITE] Actor ahead, no escape, idling")
                        return [VWIdleAction()]

                # --- After just turned, move forward if possible ---
                if self.just_turned:
                    self.just_turned = False
                    fwd = obs.get_forward()
                    forward_blocked_by_actor = (not fwd.is_empty() and fwd.or_else_raise().has_actor())
                    forward_blocked_by_wall = obs.is_wall_immediately_ahead()
                    if not forward_blocked_by_actor and not forward_blocked_by_wall:
                        print("[WHITE] Moving forward after turn")
                        return [VWMoveAction()]
                    else:
                        # If can't move forward after turn, revert turn
                        if hasattr(self, "turn_direction"):
                            opposite = VWDirection.left if self.turn_direction == VWDirection.right else VWDirection.right
                            print("[WHITE] Cannot move forward after turn, turning back")
                            return [VWTurnAction(opposite)]

                # --- Normal zigzag execution ---
                if orient != desired:
                    print(f"[WHITE] Turning to desired orientation {desired.name}")
                    return [VWTurnAction(VWDirection.right)]

                print("[WHITE] Moving forward in zigzag")
                return [VWMoveAction()]

            # --- Broadcasting dirt map ---
            if self.phase == "broadcasting":
                dirt_list: List[Dict[str, int or str]] = []
                for (dx, dy), colour in self.dirt_map.items():
                    dirt_list.append({"x": int(dx), "y": int(dy), "colour": colour})
                self.map_broadcasted = True
                self.phase = "cleaning"
                print(f"[WHITE] Broadcasting map with {len(dirt_list)} dirt locations")
                return [VWBroadcastAction(message={"dirt": dirt_list}, sender_id=self.get_own_id())]

            # --- Cleaning phase ---
            if self.phase == "cleaning":
                center = obs.get_center()
                if not center.is_empty():
                    c = center.or_else_raise()
                    if c.has_dirt():
                        coord = c.get_coord()
                        cpos = (int(coord.get_x()), int(coord.get_y()))
                        if cpos not in self.cleaned:
                            self.cleaned.add(cpos)
                            print(f"[WHITE] Cleaning dirt at {cpos}")
                            return [VWCleanAction()]

                remaining_dirt = [pos for pos in self.dirt_map.keys() if pos not in self.cleaned]
                if not remaining_dirt:
                    print("[WHITE] No remaining dirt, idling")
                    return [VWIdleAction()]

                # Move toward closest dirt
                target = min(remaining_dirt, key=lambda t: abs(int(t[0]) - x) + abs(int(t[1]) - y))
                tx, ty = int(target[0]), int(target[1])
                dx, dy = tx - x, ty - y

                if dx != 0:
                    desired = VWOrientation.east if dx > 0 else VWOrientation.west
                elif dy != 0:
                    desired = VWOrientation.south if dy > 0 else VWOrientation.north
                else:
                    desired = orient

                fwd = obs.get_forward()
                left = obs.get_left()
                right = obs.get_right()

                forward_blocked = (not fwd.is_empty() and fwd.or_else_raise().has_actor()) or obs.is_wall_immediately_ahead()
                left_free = left.is_empty() or not left.or_else_raise().has_actor()
                right_free = right.is_empty() or not right.or_else_raise().has_actor()

                if not forward_blocked:
                    print("[WHITE] Moving toward cleaning target")
                    return [VWMoveAction()]
                if forward_blocked and left_free:
                    print("[WHITE] Forward blocked, turning left toward cleaning target")
                    return [VWTurnAction(VWDirection.left)]
                if forward_blocked and right_free:
                    print("[WHITE] Forward blocked, turning right toward cleaning target")
                    return [VWTurnAction(VWDirection.right)]
                print("[WHITE] Forward blocked, cannot move, idling")
                return [VWIdleAction()]

            print("[WHITE] No specific phase action, idling")
            return [VWIdleAction()]

        except Exception as e:
            print(f"[WHITE] decide error: {e}")
            return [VWIdleAction()]


# ----------------------------
# OrangeMind & GreenMind with proper collision avoidance
# ----------------------------
class BaseCleanerMind(VWActorMindSurrogate):
    def __init__(self, colour_name: str) -> None:
        super().__init__()
        self.colour_name = colour_name.lower()
        self.map_received: bool = False
        self.targets: Set[Tuple[int, int]] = set()
        self.cleaned: Set[Tuple[int, int]] = set()

        self.just_turned: bool = False
        self.last_positions: List[Tuple[int, int]] = []     # loop detection

    def revise(self) -> None:
        try:
            pos = self.get_own_position()
            x, y = int(pos.get_x()), int(pos.get_y())
            obs = self.get_latest_observation()

            # store last 4 positions for loop detection
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
                        if self.colour_name in entry["colour"].lower():
                            self.targets.add(pos_tuple)

            # Remove cleaned targets automatically
            center = obs.get_center()
            if not center.is_empty():
                c = center.or_else_raise()
                coord = c.get_coord()
                cpos = (int(coord.get_x()), int(coord.get_y()))
                if not c.has_dirt() and cpos in self.targets:
                    self.targets.remove(cpos)
                    self.cleaned.add(cpos)

        except Exception as e:
            print(f"[{self.colour_name.upper()}] revise error: {e}")

    def decide(self) -> Iterable[VWAction]:
        try:
            pos = self.get_own_position()
            x, y = int(pos.get_x()), int(pos.get_y())
            orient = self.get_own_orientation()
            obs = self.get_latest_observation()

            # If not ready, idle
            if not self.map_received or not self.targets:
                return [VWIdleAction()]

            # --- Clean dirt if standing on it ---
            center = obs.get_center()
            if not center.is_empty():
                c = center.or_else_raise()
                if c.has_dirt():
                    coord = c.get_coord()
                    cpos = (int(coord.get_x()), int(coord.get_y()))
                    dirt_app = c.get_dirt_appearance().or_else_raise()
                    if dirt_app.get_colour().name.lower() == self.colour_name:
                        self.cleaned.add(cpos)
                        return [VWCleanAction()]

            # --- Pick nearest target ---
            target = min(self.targets, key=lambda t: abs(t[0] - x) + abs(t[1] - y))
            tx, ty = target
            dx, dy = tx - x, ty - y

            if dx != 0:
                desired = VWOrientation.east if dx > 0 else VWOrientation.west
            elif dy != 0:
                desired = VWOrientation.south if dy > 0 else VWOrientation.north
            else:
                desired = orient

            forward = obs.get_forward()
            left = obs.get_left()
            right = obs.get_right()

            forward_blocked = obs.is_wall_immediately_ahead() or \
                            (not forward.is_empty() and forward.or_else_raise().has_actor())
            left_free = left.is_empty() or not left.or_else_raise().has_actor()
            right_free = right.is_empty() or not right.or_else_raise().has_actor()

            # --- If facing wrong direction, turn toward target ---
            if orient != desired:
                self.just_turned = True
                return [VWTurnAction(VWDirection.right)]

            # --- If just turned last cycle, attempt forward ---
            if self.just_turned:
                self.just_turned = False
                if not forward_blocked:
                    return [VWMoveAction()]

            # --- Attempt forward movement ---
            if not forward_blocked:
                return [VWMoveAction()]

            # --- If blocked, try left ---
            if left_free:
                self.just_turned = True
                return [VWTurnAction(VWDirection.left)]

            # --- If blocked, try right ---
            if right_free:
                self.just_turned = True
                return [VWTurnAction(VWDirection.right)]

            # --- Fully stuck, idle ---
            return [VWIdleAction()]

        except Exception as e:
            print(f"[{self.colour_name.upper()}] decide error: {e}")
            return [VWIdleAction()]


class OrangeMind(BaseCleanerMind):
    def __init__(self) -> None:
        super().__init__("orange")


class GreenMind(BaseCleanerMind):
    def __init__(self) -> None:
        super().__init__("green")


if __name__ == "__main__":
    run(white_mind=WhiteMind(), orange_mind=OrangeMind(), green_mind=GreenMind())
