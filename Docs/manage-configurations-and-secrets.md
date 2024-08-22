# Manage Configurations and Secrets

Choreo allows you to easily manage and version your component's configurations and secrets as **file mounts** or **environment variables**.

!!! info "Note"
    All configurations and secrets applied to a Choreo component are stored in an encrypted secret vault in the cloud data plane, which is managed by WSO2.
    If you are on a private data plane, the configurations and secrets are stored in an Azure key vault or AWS secret manager attached to your data plane in your cloud environment.

## The difference between configurations and secrets

Choreo considers all configurations and secrets to be sensitive content when storing them, but gives you the option to choose between secret or configuration when you create a file mount or an environment variable.

- **Secrets** are write-only. Once you create a secret, you cannot see or retrieve its content via the Choreo Console. However, you can overwrite the existing content at any time.
- **Configurations** can be read and updated via the Choreo Console after you create them.
  
    !!!info "Note"

          If you want to include sensitive data such as database passwords, cloud credentials, service accounts, and so on, the recommended approach is to use a secret instead of a configuration.

## Apply a file mount to your container

Follow these steps to apply a file mount to a component you have created:

1. Sign in to the [Choreo Console](https://console.choreo.dev/).
2. In the **Component Listing** pane, click on the component for which you want to define configurations and secrets.
3. In the left navigation menu, click **DevOps** and then click **Configs & Secrets**.
4. Click **+ Create**.
5. In the **Mount a Configuration** pane, do the following:
    1. Select a **Config Type** depending on your requirement.
    2. Select **File Mount** as the **Mount Type**.
    3. Click **Next**.
    4. In the **Config Name** field, specify a name for the file mount.
  
        !!!tip

                The configuration name does not affect the file mount or its content. It is only a reference to identify the configuration or secret you create.

    5. In the **Mount Path** field, specify where to mount the file inside the container. Use an absolute file path with the file name and extension if applicable.
  
        !!!tip

                The file name in the mount path does not need to match the configuration name or the name of the file you upload.

    6. Upload a configuration file or copy and paste the configuration content into the editor.

6. Click **Create**.

    ![Create file mount](../assets/img/devops-and-ci-cd/configs/create-file-mount.png){.cInlineImage-full}
  
    !!!info "Note"
           
            Configurations and secrets are applied immediately to your environment on creation. To ensure that the container reflects the new content, your existing running replicas undergo a rolling restart.

## Apply environment variables to your container

Follow these steps to apply environment variables to a component you have created:

1. Sign in to the [Choreo Console](https://console.choreo.dev/).
2. In the **Component Listing** pane, click on the component for which you want to define configurations and secrets.
3. In the left navigation menu, click **DevOps** and then click **Configs & Secrets**.
4. Click **+ Create**.
5. In the **Mount a Configuration** pane, do the following:
    1. Select a **Config Type** depending on your requirement.
    2. Select **File Mount** as the **Mount Type**.
    3. Click **Next**.
    4. In the **Config Name** field, specify a name for the file mount.
  
        !!!tip

                The configuration name you specify does not affect the environment variables you set. It is only a reference to identify the configuration or secret you create.

    5. Add the necessary environment variables as key-value pairs. You can click **Add Item** to add any number of environment variables.

6. Click **Create**.
   
    ![Set environment variables](../assets/img/devops-and-ci-cd/configs/create-env-vars.png){.cInlineImage-full}

## Update an existing configuration or secret

Follow these steps to update a configuration or secret you have defined:

1. Sign in to the [Choreo Console](https://console.choreo.dev/).
2. In the **Component Listing** pane, click on the component for which you want to update configurations and secrets.
3. In the left navigation menu, click **DevOps** and then click **Configs & Secrets**.
4. Click the edit icon corresponding to the configuration or secret you want to update.
5. Apply the necessary changes and click **Save**.

    ![Modify existing configs](../assets/img/devops-and-ci-cd/configs/create-or-delete-config.png){.cInlineImage-half}

## Delete an existing configuration or secret

Follow these steps to delete a configuration or secret you have defined:

1. Sign in to the [Choreo Console](https://console.choreo.dev/).
2. In the **Component Listing** pane, click on the component for which you want to delete configurations and secrets.
3. In the left navigation menu, click **DevOps** and then click **Configs & Secrets**.
4. Click the delete icon corresponding to the configuration or secret you want to delete.
5. Enter the name of the configuration or secret to confirm deletion.
6. Click **Delete**.

## Manage Ballerina configurables

Choreo manages the [Ballerina configurables](https://ballerina.io/learn/by-example/configurable-variables/) for the Ballerina components you create.

When you deploy or promote a Ballerina application, you can modify the Ballerina configurables via the **Deploy** page.
  
!!!tip

      You can use configurables instead of environment variables to add file mounts to a Ballerina component.
      Environment variables are primarily for components written in other languages.
