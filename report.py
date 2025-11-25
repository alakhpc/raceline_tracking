from dataclasses import dataclass
from typing import Optional


@dataclass
class LapReport:
    """Report containing lap statistics and results."""

    completed: bool
    lap_time: float
    track_violations: int
    avg_speed: float
    max_speed: float
    min_speed: float
    distance_traveled: float
    steps: int
    wall_time: float
    dnf_reason: Optional[str] = None

    @property
    def penalty_time(self) -> float:
        """Calculate penalty time (5s per track violation)."""
        return self.track_violations * 5.0

    @property
    def total_score(self) -> float:
        """Calculate total score (lap time + penalties)."""
        return self.lap_time + self.penalty_time

    @property
    def speedup(self) -> float:
        """Calculate simulation speedup vs real-time."""
        return self.lap_time / self.wall_time if self.wall_time > 0 else 0.0

    def print(self):
        """Print a formatted lap report."""
        print("\n" + "=" * 60)
        print("                    LAP REPORT")
        print("=" * 60)

        status = "✓ COMPLETED" if self.completed else "✗ DNF (Did Not Finish)"
        print(f"\n  Status:              {status}")

        if self.completed:
            print(f"  Lap Time:            {self.lap_time:.3f}s")
        else:
            print(f"  Time Elapsed:        {self.lap_time:.3f}s")
            print(f"  DNF Reason:          {self.dnf_reason or 'Unknown'}")

        print(f"\n  Track Violations:    {self.track_violations}")

        print(f"\n  Distance Traveled:   {self.distance_traveled:.1f}m")
        print(f"  Average Speed:       {self.avg_speed:.2f} m/s ({self.avg_speed * 3.6:.1f} km/h)")
        print(f"  Max Speed:           {self.max_speed:.2f} m/s ({self.max_speed * 3.6:.1f} km/h)")
        print(f"  Min Speed:           {self.min_speed:.2f} m/s ({self.min_speed * 3.6:.1f} km/h)")

        print(f"\n  Simulation Steps:    {self.steps}")
        print(f"  Wall Clock Time:     {self.wall_time:.3f}s")
        print(f"  Speedup:             {self.speedup:.1f}x real-time")

        print("\n" + "=" * 60)

        if self.completed:
            print("\n  SCORING:")
            print(f"    Raw Lap Time:      {self.lap_time:.3f}s")
            print(f"    Violation Penalty: +{self.penalty_time:.1f}s ({self.track_violations} x 5s)")
            print(f"    Total Score:       {self.total_score:.3f}s")
            print("=" * 60 + "\n")
