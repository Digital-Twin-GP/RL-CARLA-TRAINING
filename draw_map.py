import carla
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class MapVisualization:
    def __init__(self, args):
        self.carla_client = carla.Client(args.host, args.port, worker_threads=1)
        self.world = self.carla_client.get_world()
        self.map = self.world.get_map()
        self.fig, self.ax = plt.subplots()
        self.all_x = []
        self.all_y = []

    @staticmethod
    def lateral_shift(transform, shift):
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    # ...existing code...

    def draw_and_fill_roads(self):
        precision = 0.1
        topology = self.map.get_topology()
        topology = [x[0] for x in topology]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        set_waypoints = []
        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            set_waypoints.append(waypoints)
        for waypoints in set_waypoints:
            road_left_side = [self.lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            road_right_side = [self.lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
            if len(road_left_side) > 2 and len(road_right_side) > 2:
                poly_x = [p.x for p in road_left_side] + [p.x for p in reversed(road_right_side)]
                poly_y = [-p.y for p in road_left_side] + [-p.y for p in reversed(road_right_side)]
                polygon = Polygon(list(zip(poly_x, poly_y)), closed=True, facecolor='lightgray', edgecolor='none', alpha=0.8)
                self.ax.add_patch(polygon)
                # Thicker lines: linewidth=2.5
                self.ax.plot([p.x for p in road_left_side], [-p.y for p in road_left_side], color='darkslategrey', linewidth=1.4)
                self.ax.plot([p.x for p in road_right_side], [-p.y for p in road_right_side], color='darkslategrey', linewidth=1.4)
                self.all_x.extend(poly_x)
                self.all_y.extend(poly_y)

# ...existing code...

    def save_image(self, filename="carla_map_filled.png"):
        self.ax.axis('equal')
        self.ax.axis('off')
        plt.tight_layout(pad=0)
        self.fig.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Map image saved to {filename}")

    def print_bounds(self):
        min_x, max_x = min(self.all_x), max(self.all_x)
        min_y, max_y = min(self.all_y), max(self.all_y)
        print("Image/Map bounds (for pixel-to-location mapping):")
        print(f"  Leftmost (min x): {min_x}")
        print(f"  Rightmost (max x): {max_x}")
        print(f"  Uppermost (max y): {max_y}")
        print(f"  Downmost (min y): {min_y}")

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--host', default='localhost', help='CARLA host')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='CARLA port')
    args = argparser.parse_args()

    viz = MapVisualization(args)
    viz.draw_and_fill_roads()
    viz.print_bounds()
    viz.save_image("carla_map_filled.png")

if __name__ == "__main__":
    main()