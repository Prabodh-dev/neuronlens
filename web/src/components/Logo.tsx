/** A neuron seen through a lens. */
export function Logo({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="14.25" fill="none" stroke="#fff" strokeWidth="1.5" />
      <path
        d="M16 16 L9 9 M16 16 L23.5 10 M16 16 L22 23.5 M16 16 L9.5 22.5 M9 9 L6.5 10.5 M9 9 L9.5 6 M23.5 10 L26 11 M22 23.5 L24.5 25"
        stroke="#fff"
        strokeWidth="1.3"
        strokeLinecap="round"
        opacity="0.55"
      />
      <circle cx="16" cy="16" r="3.6" fill="#fff" />
    </svg>
  );
}
