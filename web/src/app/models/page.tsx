"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { ModelPicker } from "@/components/landing/ModelPicker";
import { useModels } from "@/components/ModelProvider";

/** The model selector, reachable from inside any model (Models in the top bar). With fewer than two
 *  models answering there is nothing to choose, so it goes to the home page. */
export default function Models() {
  const { current, probes } = useModels();
  const pending = current === null || probes.some((p) => p.health === undefined);
  const choices = probes.filter((p) => p.health).length;
  const router = useRouter();
  useEffect(() => {
    if (!pending && choices < 2) router.replace("/");
  }, [pending, choices, router]);
  if (pending || choices < 2) return <div aria-hidden="true" style={{ minHeight: "100vh" }} />;
  return <ModelPicker currentId={current.id} />;
}
