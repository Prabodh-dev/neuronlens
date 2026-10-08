import Link from "next/link";
import { EmptyState } from "@/components/States";

export default function NotFound() {
  return (
    <div className="page">
      <div className="card">
        <EmptyState title="This page doesn't exist" action={<Link className="btn" href="/">Go home</Link>}>
          The address may be mistyped, or the page was moved.
        </EmptyState>
      </div>
    </div>
  );
}
