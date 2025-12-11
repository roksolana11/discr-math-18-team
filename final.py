"""
Module for finding the shortest path on a rectangular terrain with elevation data.

Uses Dijkstra's algorithm to find the optimal route between two points on a height map.
Supports result visualization and search process animation.

Command line usage examples:
    python pathfinder.py -f map.txt -s 0,0 -e 10,10
    python pathfinder.py -f map.txt -s 0,0 -e 10,10 -v -o result
    python pathfinder.py -f map.txt -s 0,0 -e 10,10 -a -o animation
"""

import argparse
import heapq
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

OBSTACLE_MARKERS = ['-1', 'X', 'x', '#', '*']


def arguments_pars():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Object with arguments:
            - file (str): Path to height map file
            - start (str): Start point in "row,col" format
            - end (str): End point in "row,col" format
            - step (float): Horizontal distance between adjacent points
            - animate (bool): Whether to create GIF animation
            - visualise (bool): Whether to create PNG image
            - output (str): Output filename (without extension)
            - fps (int): Frames per second for animation
    """

    parser = argparse.ArgumentParser(description='Пошук найкоротшого шляху на прямокутній ділянці.')
    
    parser.add_argument('-f', '--file', required=True, help='CSV файл з матрицею висот.')
    parser.add_argument('-s', '--start', required=True, help='Стартова точка (рядок,стовпець)')
    parser.add_argument('-e', '--end', required=True, help='Кінцева точка (рядок,стовпець)')
    parser.add_argument('--step', type=float, default=1.0, help='Відстань між точками')

    parser.add_argument('-a', '--animate', action='store_true', help='Створити GIF анімацію пошуку')
    parser.add_argument('-v', '--visualise', action='store_true', help='Створити зображення результату')
    parser.add_argument('-o', '--output', type=str, default='result', help='Назва вихідного файлу (без розширення)')
    parser.add_argument('--fps', type=int, default=10, help='Кадрів на секунду для анімації')

    return parser.parse_args()


def read_matrix(path: str) -> list[list[int | None]]:
    """
    Read height matrix from a text file.

    File format: space-separated numbers, each matrix row on a new line.
    Obstacles are marked with: -1, X, x, #, *

    Args:
        path: Path to the matrix file.

    Returns:
        2D list with heights (int) or None for obstacles.

    Raises:
        FileNotFoundError: If file not found.
        ValueError: If file contains invalid data.
        PermissionError: If no access to file.

    Examples:
        >>> import tempfile, os
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        ...     _ = f.write("1 2 3\\n4 5 6\\n7 8 9")
        ...     fname = f.name
        >>> read_matrix(fname)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> os.unlink(fname)

        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        ...     _ = f.write("1 X 3\\n4 5 6")
        ...     fname = f.name
        >>> read_matrix(fname)
        [[1, None, 3], [4, 5, 6]]
        >>> os.unlink(fname)

        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        ...     _ = f.write("10.5 20.7\\n30 40")
        ...     fname = f.name
        >>> read_matrix(fname)
        [[10, 20], [30, 40]]
        >>> os.unlink(fname)
    """
    matrix = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                row = []
                for el in line.strip().split():
                    if el in OBSTACLE_MARKERS:
                        row.append(None)
                    else:
                        try:
                            row.append(int(el))
                        except ValueError:
                            try:
                                row.append(int(float(el)))
                            except ValueError:
                                raise ValueError(f"Invalid value: '{el}'")
                matrix.append(row)
    return matrix


def coords_pars(coord: str) -> tuple[int, int]:
    """
    Parse coordinates from string in "x,y" format.

    Args:
        coord: String with coordinates like "5,10" or "5, 10".

    Returns:
        Tuple (row, col) with integer coordinates.

    Raises:
        ValueError: If format is invalid or values are not integers.

    Examples:
        >>> coords_pars("0,0")
        (0, 0)
        >>> coords_pars("5,10")
        (5, 10)
        >>> coords_pars("  3 , 7  ")
        (3, 7)
        >>> coords_pars("12,34")
        (12, 34)
        >>> coords_pars("abc")
        Traceback (most recent call last):
            ...
        ValueError: Expected format 'x,y', got: 'abc'
        >>> coords_pars("1,2,3")
        Traceback (most recent call last):
            ...
        ValueError: Expected format 'x,y', got: '1,2,3'
    """
    parts = coord.strip().replace(' ', '').split(',')
    if len(parts) != 2:
        raise ValueError(f"Expected format 'x,y', got: '{coord}'")
    x, y = int(parts[0]), int(parts[1])
    return x, y


def matrix_to_array(matrix: list[list[int | None]]) -> np.ndarray:
    """
    Convert matrix with None values to numpy array with np.nan.

    Used for matplotlib visualization which cannot handle None values.

    Args:
        matrix: 2D list with integers and None (obstacles).

    Returns:
        numpy.ndarray with float values, None replaced with np.nan.

    Examples:
        >>> m = [[1, 2], [3, 4]]
        >>> arr = matrix_to_array(m)
        >>> arr.tolist()
        [[1.0, 2.0], [3.0, 4.0]]

        >>> m = [[1, None], [None, 4]]
        >>> arr = matrix_to_array(m)
        >>> bool(np.isnan(arr[0][1]))
        True
        >>> bool(np.isnan(arr[1][0]))
        True
        >>> float(arr[1][1])
        4.0
    """
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    arr = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] is None:
                arr[i][j] = np.nan
            else:
                arr[i][j] = matrix[i][j]
    return arr


def dijkstra(matrix: list[list[int | None]],
             start: tuple[int, int],
             end: tuple[int, int],
             step: float,
             record_frames: bool = False) -> tuple:
    """
    Find shortest path using Dijkstra's algorithm with elevation.

    The path cost between adjacent cells is calculated as:
        cost = sqrt(height_difference^2 + step^2)

    This accounts for both horizontal distance and elevation change.

    Args:
        matrix: 2D height map with None for obstacles.
        start: Starting point as (row, col).
        end: Ending point as (row, col).
        step: Horizontal distance between adjacent cells.
        record_frames: If True, record frames for animation.

    Returns:
        If record_frames=False:
            tuple: (distance, path) where path is list of (row, col).
        If record_frames=True:
            tuple: (distance, path, frames) for animation.

        Returns (inf, [], []) if path not found.

    Examples:
        >>> m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> dist, path = dijkstra(m, (0, 0), (2, 2), 1.0)
        >>> len(path) > 0
        True
        >>> path[0]
        (0, 0)
        >>> path[-1]
        (2, 2)

        >>> dist, path = dijkstra(m, (0, 0), (0, 0), 1.0)
        >>> dist
        0
        >>> path
        [(0, 0)]
    """
    row_num = len(matrix)
    cols_num = len(matrix[0]) if matrix else 0

    if row_num == 0 or cols_num == 0:
        print("Matrix is empty")
        if record_frames:
            return float('inf'), [], []
        return float('inf'), []

    if matrix[start[0]][start[1]] is None:
        print("Start point is an obstacle")
        if record_frames:
            return float('inf'), [], []
        return float('inf'), []

    if matrix[end[0]][end[1]] is None:
        print("End point is an obstacle")
        if record_frames:
            return float('inf'), [], []
        return float('inf'), []

    if start == end:
        if record_frames:
            return 0, [start], []
        return 0, [start]

    distance_table = [[float('inf')] * cols_num for _ in range(row_num)]
    distance_table[start[0]][start[1]] = 0

    visited = set()
    prev_coords = [[None] * cols_num for _ in range(row_num)]

    queue = [(0, start[0], start[1])]
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    frames = [] if record_frames else None
    if record_frames:
        total_cells = row_num * cols_num
        if total_cells < 500:
            frame_interval = 1
        elif total_cells < 2000:
            frame_interval = max(1, total_cells // 300)
        else:
            frame_interval = max(1, total_cells // 200)
        step_count = 0

    while queue:
        distance, row, col = heapq.heappop(queue)

        if (row, col) in visited:
            continue

        visited.add((row, col))

        if record_frames:
            step_count += 1
            if step_count % frame_interval == 0 or (row, col) == end:
                frames.append({
                    'visited': set(visited),
                    'current': (row, col),
                    'queue': [(r, c) for _, r, c in queue[:50]]
                })

        if (row, col) == end:
            break

        for r, c in directions:
            new_row = row + r
            new_col = col + c

            if not (0 <= new_row < row_num and 0 <= new_col < cols_num):
                continue

            if (new_row, new_col) in visited:
                continue

            if matrix[new_row][new_col] is None:
                continue

            height_difference = matrix[row][col] - matrix[new_row][new_col]
            weight = math.sqrt(height_difference ** 2 + step ** 2)

            new_distance = distance + weight

            if new_distance < distance_table[new_row][new_col]:
                distance_table[new_row][new_col] = new_distance
                prev_coords[new_row][new_col] = (row, col)
                heapq.heappush(queue, (new_distance, new_row, new_col))

    if distance_table[end[0]][end[1]] == float('inf'):
        print("Path not found (end point unreachable)")
        if record_frames:
            return float('inf'), [], frames if frames else []
        return float('inf'), []

    path = []
    current = end

    while current is not None:
        path.append(current)
        current = prev_coords[current[0]][current[1]]
    path.reverse()

    if not path or path[0] != start:
        print("Failed to reconstruct path")
        if record_frames:
            return float('inf'), [], frames if frames else []
        return float('inf'), []

    if record_frames:
        return distance_table[end[0]][end[1]], path, frames
    else:
        return distance_table[end[0]][end[1]], path


def create_visualisation(matrix: list[list[int | None]],
                         start: tuple[int, int],
                         end: tuple[int, int],
                         path: list[tuple[int, int]],
                         distance: float,
                         output_path: str) -> str:
    """
    Create a static PNG image of the pathfinding result.

    Shows terrain with color-coded elevation, start/end markers,
    and the found path drawn in white.

    Args:
        matrix: 2D height map.
        start: Starting point (row, col).
        end: Ending point (row, col).
        path: List of points forming the path.
        distance: Total path distance.
        output_path: Output filename without extension.

    Returns:
        Full path to saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    matrix_arr = matrix_to_array(matrix)
    terrain = ax.imshow(matrix_arr, cmap='terrain', aspect='equal')
    plt.colorbar(terrain, ax=ax, label='Elevation')

    if path and len(path) > 1:
        path_cols = [p[1] for p in path]
        path_rows = [p[0] for p in path]
        ax.plot(path_cols, path_rows, 'white', linewidth=2, label='Path')

    ax.scatter([start[1]], [start[0]], c='lime', s=150, marker='o',
               edgecolors='black', linewidths=2, label='Start', zorder=100)
    ax.scatter([end[1]], [end[0]], c='red', s=150, marker='o',
               edgecolors='black', linewidths=2, label='End', zorder=100)

    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'Shortest Path. Distance: {distance:.2f}, Steps: {len(path)}')

    filepath = f"{output_path}.png"
    try:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Image saved: {filepath}")
    except PermissionError:
        print(f"No permission to save file: {filepath}")
    except Exception:
        print("Error saving image.")
    finally:
        plt.close()
    return filepath


def create_search_animation(matrix: list[list[int | None]],
                            start: tuple[int, int],
                            end: tuple[int, int],
                            step: float = 1.0,
                            output_path: str = 'pathfinding',
                            fps: int = 10,
                            show_live: bool = True) -> str | None:
    """
    Create an animated GIF showing the pathfinding process.

    The animation shows:
    - Blue dots: visited cells
    - Yellow dots: cells in the queue
    - Red star: current cell being processed
    - White line: final path (at the end)

    Args:
        matrix: 2D height map.
        start: Starting point (row, col).
        end: Ending point (row, col).
        step: Horizontal distance between adjacent points.
        output_path: Output filename without extension.
        fps: Frames per second.
        show_live: Whether to show animation window.

    Returns:
        Path to saved GIF file, or None if failed.
    """
    print("Running algorithm with animation recording...")
    distance, path, frames = dijkstra(matrix, start, end, step, record_frames=True)

    if distance == float('inf'):
        print("Path not found!")
        return None

    if not frames:
        print("Failed to record animation frames")
        return None

    print(f"Path found! Distance: {distance:.2f}")
    print(f"Creating animation ({len(frames)} frames)...")

    fig, ax = plt.subplots(figsize=(10, 8))

    matrix_arr = matrix_to_array(matrix)
    terrain = ax.imshow(matrix_arr, cmap='terrain', aspect='equal')
    plt.colorbar(terrain, ax=ax, label='Elevation')

    visited_scatter = ax.scatter([], [], c='blue', s=3, alpha=0.4, label='Visited', zorder=2)
    queue_scatter = ax.scatter([], [], c='yellow', s=5, alpha=0.6, label='In queue', zorder=3)
    current_scatter = ax.scatter([], [], c='red', s=100, marker='*', label='Current', zorder=4)
    path_line, = ax.plot([], [], 'white', linewidth=2, label='Path', zorder=5)

    ax.scatter([start[1]], [start[0]], c='lime', s=150, marker='o',
               edgecolors='black', linewidths=2, label='Start', zorder=100)
    ax.scatter([end[1]], [end[0]], c='red', s=150, marker='o',
               edgecolors='black', linewidths=2, label='End', zorder=100)

    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Shortest Path Search (Dijkstra\'s Algorithm)')

    def init():
        visited_scatter.set_offsets(np.empty((0, 2)))
        queue_scatter.set_offsets(np.empty((0, 2)))
        current_scatter.set_offsets(np.empty((0, 2)))
        path_line.set_data([], [])
        return visited_scatter, queue_scatter, current_scatter, path_line

    def animate(frame_idx):
        if frame_idx < len(frames):
            frame = frames[frame_idx]

            if frame['visited']:
                visited_coords = np.array([[c, r] for r, c in frame['visited']])
                visited_scatter.set_offsets(visited_coords)
            else:
                visited_scatter.set_offsets(np.empty((0, 2)))

            if frame['queue']:
                queue_coords = np.array([[c, r] for r, c in frame['queue']])
                queue_scatter.set_offsets(queue_coords)
            else:
                queue_scatter.set_offsets(np.empty((0, 2)))

            current_scatter.set_offsets([[frame['current'][1], frame['current'][0]]])
            ax.set_title(f'Searching... Step {frame_idx + 1}/{len(frames)}')

        else:
            if path and len(path) > 1:
                path_cols = [p[1] for p in path]
                path_rows = [p[0] for p in path]
                path_line.set_data(path_cols, path_rows)
            ax.set_title(f'Path found! Distance: {distance:.2f}, Length: {len(path)} steps',
                        fontsize=14, fontweight='bold', color='green')

        return visited_scatter, queue_scatter, current_scatter, path_line

    total_frames = len(frames) + 15

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000//max(1, fps), blit=True
    )

    if show_live:
        print("Showing animation... (close window to continue)")
        try:
            plt.show()
        except Exception:
            print("Could not show animation.")

    filepath = f"{output_path}.gif"
    print(f"Saving to {filepath}...")
    try:
        anim.save(filepath, writer='pillow', fps=max(1, fps))
        print(f"Animation saved: {filepath}")
        plt.close()
        return filepath
    except PermissionError:
        print(f"No permission to save file: {filepath}")
    except ImportError:
        print("Library 'pillow' not installed. Install with: pip install pillow")
    except Exception as e:
        print(f"Error saving animation: {e}")
    finally:
        plt.close()
    return None


def validate_matrix(matrix: list[list[int | None]]) -> tuple[bool, str]:
    """
    Validate that matrix is properly formed.

    Args:
        matrix: 2D list to validate.

    Returns:
        Tuple (is_valid, error_message).

    Examples:
        >>> validate_matrix([[1, 2], [3, 4]])
        (True, '')
        >>> validate_matrix([])
        (False, 'Matrix is empty')
        >>> validate_matrix([[]])
        (False, 'First row is empty')
        >>> validate_matrix([[1, 2], [3]])
        (False, 'Row 2 has 1 columns, expected 2')
    """
    if not matrix:
        return False, "Matrix is empty"

    if not matrix[0]:
        return False, "First row is empty"

    cols_count = len(matrix[0])
    for i, row in enumerate(matrix):
        if len(row) != cols_count:
            return False, f"Row {i+1} has {len(row)} columns, expected {cols_count}"

    return True, ""


def validate_coordinates(coord: tuple[int, int],
                         rows: int,
                         cols: int,
                         name: str = "Point") -> tuple[bool, str]:
    """
    Validate that coordinates are within matrix bounds.

    Args:
        coord: (row, col) tuple to validate.
        rows: Number of rows in matrix.
        cols: Number of columns in matrix.
        name: Name for error messages (e.g., "Start", "End").

    Returns:
        Tuple (is_valid, error_message).

    Examples:
        >>> validate_coordinates((0, 0), 10, 10)
        (True, '')
        >>> validate_coordinates((5, 5), 10, 10)
        (True, '')
        >>> validate_coordinates((-1, 0), 10, 10)
        (False, 'Point has negative coordinates: (-1, 0)')
        >>> validate_coordinates((10, 5), 10, 10, "Start")
        (False, 'Start (10, 5) is outside matrix bounds (10x10)')
    """
    if coord[0] < 0 or coord[1] < 0:
        return False, f"{name} has negative coordinates: {coord}"

    if coord[0] >= rows or coord[1] >= cols:
        return False, f"{name} {coord} is outside matrix bounds ({rows}x{cols})"

    return True, ""


def main():
    """
    Main entry point for command line usage.

    Parses arguments, reads matrix, validates inputs,
    and runs pathfinding with optional visualization.
    """
    args = arguments_pars()

    try:
        matrix = read_matrix(args.file)
    except FileNotFoundError:
        print(f"File not found: '{args.file}'")
        return
    except PermissionError:
        print(f"No access to file: '{args.file}'")
        return
    except IsADirectoryError:
        print(f"'{args.file}' is a directory, not a file")
        return
    except UnicodeDecodeError:
        print("Invalid file encoding. Use UTF-8")
        return
    except ValueError:
        print("Invalid data format in file.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    is_valid, error = validate_matrix(matrix)
    if not is_valid:
        print(error)
        return

    rows_num = len(matrix)
    cols_num = len(matrix[0])

    step = args.step

    if step <= 0:
        print(f"Step must be positive, got: {step}")
        return

    if math.isnan(step) or math.isinf(step):
        print(f"Step must be a finite number, got: {step}")
        return

    try:
        start = coords_pars(args.start)
    except ValueError as e:
        print(f"Invalid start point format: '{args.start}'. {e}")
        return

    try:
        end = coords_pars(args.end)
    except ValueError as e:
        print(f"Invalid end point format: '{args.end}'. {e}")
        return

    is_valid, error = validate_coordinates(start, rows_num, cols_num, "Start point")
    if not is_valid:
        print(error)
        return

    is_valid, error = validate_coordinates(end, rows_num, cols_num, "End point")
    if not is_valid:
        print(error)
        return

    if args.fps <= 0:
        print(f"FPS must be positive, got: {args.fps}")
        return

    if args.animate:
        create_search_animation(matrix, start, end, step, output_path=args.output, fps=args.fps)
        return

    distance, path = dijkstra(matrix, start, end, step, record_frames=False)

    if distance == float('inf'):
        return

    print(f"Distance: {distance:.4f}")
    print(f"Path length: {len(path)} steps")
    print(f"Path: {path}")

    if args.visualise:
        create_visualisation(matrix, start, end, path, distance, args.output)


if __name__ == "__main__":
    main()
